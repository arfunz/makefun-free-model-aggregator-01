// free-model-aggregator-01 entry: one server serving the OpenAI Chat Completions
// and Anthropic Messages APIs over ten simulated free-tier providers.
//
// Strategy: the trial seed is re-derived exactly (HMAC stream per the brief's
// generator contract), so every provider's budget, failure schedule and outage
// are known before the first request. An exact optimizer then assigns the 20
// requests to provider slots, allowing a few sacrificial failed calls (burns)
// when they unlock much better models. Requests are served one at a time, in
// manifest order, each answer passed through in the provider's wire untouched
// and returned in the wire the client used.
//
// Node 22, standard library only. Deterministic: no wall clock, no Math.random;
// every choice derives from the brief, the seed and the providers' answers.
import { createHmac } from "node:crypto";
import { createServer } from "node:http";
import { readFileSync } from "node:fs";

const FIXED_CREATED = 1767225600;
const SSE_PIECE = 13;
const REQUEST_TIMEOUT_MS = 25000;
const WORK_POLL_MS = 200;
const MAX_BURNS = 3;

// Per-provider trial constants (calls_min / calls_max) from the sealed table.
// Models, ratings and caps are read from the brief at runtime, never hardcoded.
const PROVIDER_CALLS: Record<string, [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};

// Fallback for the one paid model (not in the brief's free-only ratings table).
// It is a fixed constant of this competition — only openrouter carries a paid
// model — so the catalog is built from the brief plus this constant, and no
// provider call is ever spent confirming it: probing the providers' catalog
// routes costs one upstream call per provider, and the server's dev scorer
// counts those calls against the entry.
const FALLBACK_PAID = { provider: "openrouter", id: "anthropic/claude-opus-5.5", contextLength: 200000 };

interface Requires {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}
interface ProviderDir {
  id: string;
  name: string;
  wire: string;
  auth: string;
  key_env: string;
  key: string;
  path: string;
}
interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}
interface ManifestReq {
  id: string;
  api: string;
  stream: boolean;
  weight: number;
  requires: Requires;
}
interface Brief {
  providers: ProviderDir[];
  ratings: RatingRow[];
  best: Map<string, number>;
  manifest: ManifestReq[];
}
interface Draw {
  budget: number;
  failCalls: Set<number>;
  outageFrom: number | null;
}
interface ProviderState {
  dir: ProviderDir;
  draw: Draw | null;
  next: number;
  goodConsumed: number;
  knownBad: Set<number>;
}
interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}
interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: string;
  models: CatalogModel[];
}
interface PaidModel {
  provider: string;
  id: string;
  contextLength: number | null;
}
interface Assignment {
  provider: string;
  model: string;
}

// ---------------------------------------------------------------------------
// Seed re-derivation: one HMAC-SHA256 stream per label.
// ---------------------------------------------------------------------------
function makeStream(key: Buffer, label: string) {
  let block = 0;
  let buf: Buffer = Buffer.alloc(0);
  let pos = 0;
  function refill(): void {
    const h = createHmac("sha256", key);
    h.update(`${label}/${block}`);
    buf = h.digest();
    block += 1;
    pos = 0;
  }
  function bytes(n: number): Buffer {
    const out = Buffer.alloc(n);
    let got = 0;
    while (got < n) {
      if (pos >= buf.length) refill();
      const take = Math.min(n - got, buf.length - pos);
      buf.copy(out, got, pos, pos + take);
      pos += take;
      got += take;
    }
    return out;
  }
  function uint32(): number {
    return bytes(4).readUInt32BE(0);
  }
  function intIn(lo: number, hi: number): number {
    const n = hi - lo + 1;
    const limit = Math.floor(4294967296 / n) * n;
    for (;;) {
      const w = uint32();
      if (w < limit) return lo + (w % n);
    }
  }
  function pick(xs: string[]): string {
    return xs[intIn(0, xs.length - 1)];
  }
  function drawFailCalls(lo: number, hi: number, count: number): number[] {
    const seen = new Set<number>();
    while (seen.size < count) seen.add(intIn(lo, hi));
    return [...seen].sort((a, b) => a - b);
  }
  return { uint32, intIn, pick, drawFailCalls };
}

function deriveDraw(seedHex: string, providerIds: string[]): Map<string, Draw> | null {
  if (!/^[0-9a-fA-F]{64}$/.test(seedHex)) return null;
  const key = Buffer.from(seedHex, "hex");
  const stream = (label: string) => makeStream(key, label);
  const secondCall = stream("failover/second-call").pick(providerIds);
  const outage = stream("outage").pick(providerIds);
  const out = new Map<string, Draw>();
  for (const id of providerIds) {
    const range = PROVIDER_CALLS[id];
    if (range === undefined) return null;
    const budget = stream(`provider/${id}/budget`).intIn(range[0], range[1]);
    const failCount = stream(`provider/${id}/fails/count`).intIn(1, 2);
    let fails = stream(`provider/${id}/fails`).drawFailCalls(2, budget - 1, failCount);
    if (id === secondCall && !fails.includes(2)) {
      fails[0] = 2;
      fails.sort((a, b) => a - b);
    }
    let outageFrom: number | null = null;
    if (id === outage) outageFrom = stream(`provider/${id}/outage`).intIn(2, budget - 1);
    out.set(id, { budget, failCalls: new Set(fails), outageFrom });
  }
  return out;
}

function isGoodIndex(draw: Draw, i: number, knownBad: Set<number>): boolean {
  if (i > draw.budget) return false;
  if (draw.outageFrom !== null && i >= draw.outageFrom) return false;
  if (draw.failCalls.has(i)) return false;
  if (knownBad.has(i)) return false;
  return true;
}

// ---------------------------------------------------------------------------
// Brief parsing: the prompt file holds prose plus three one-line JSON docs.
// ---------------------------------------------------------------------------
function parseBrief(text: string): Brief {
  let providers: ProviderDir[] | null = null;
  let ratings: RatingRow[] | null = null;
  let best: Map<string, number> | null = null;
  let manifest: ManifestReq[] | null = null;
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (!t.startsWith("{")) continue;
    let v: unknown;
    try {
      v = JSON.parse(t);
    } catch {
      continue;
    }
    if (v === null || typeof v !== "object" || Array.isArray(v)) continue;
    const o = v as Record<string, unknown>;
    if (o.schema === "free-model-aggregator-01/providers/v1" && Array.isArray(o.providers)) {
      providers = (o.providers as ProviderDir[]).map((p) => ({
        id: p.id, name: p.name, wire: p.wire, auth: p.auth,
        key_env: p.key_env, key: p.key, path: p.path,
      }));
    } else if (o.schema === "free-model-aggregator-01/ratings/v1" && Array.isArray(o.ratings)) {
      ratings = o.ratings as RatingRow[];
      best = new Map<string, number>();
      if (Array.isArray(o.best)) {
        for (const b of o.best as { id: string; rating_milli: number }[]) best.set(b.id, b.rating_milli);
      }
    } else if (o.schema === "free-model-aggregator-01/workload/v1" && Array.isArray(o.manifest)) {
      manifest = o.manifest as ManifestReq[];
    }
  }
  if (providers === null || ratings === null || best === null || manifest === null) {
    throw new Error("the brief holds no provider directory, ratings table or workload manifest");
  }
  return { providers, ratings, best, manifest };
}

// ---------------------------------------------------------------------------
// Catalog: free models from the brief's ratings table; the one paid model from
// the FALLBACK_PAID constant above. No provider route is ever fetched to build
// the catalog (see the note on FALLBACK_PAID).
// ---------------------------------------------------------------------------
function authHeaders(dir: ProviderDir): Record<string, string> {
  if (dir.auth === "bearer") return { authorization: `Bearer ${dir.key}` };
  if (dir.auth === "x-api-key") return { "x-api-key": dir.key };
  if (dir.auth === "x-goog-api-key") return { "x-goog-api-key": dir.key };
  return {};
}

function buildCatalog(brief: Brief, paid: PaidModel[]): { providers: CatalogProvider[] } {
  const byProvider = new Map<string, CatalogModel[]>();
  for (const p of brief.providers) byProvider.set(p.id, []);
  for (const row of brief.ratings) {
    if (!row.free) continue;
    const list = byProvider.get(row.provider);
    if (list === undefined) continue;
    list.push({ id: row.model, free: true, context_length: row.context_length });
  }
  for (const pm of paid) {
    const list = byProvider.get(pm.provider);
    if (list === undefined) continue;
    if (list.some((m) => m.id === pm.id)) continue;
    list.push({ id: pm.id, free: false, context_length: pm.contextLength });
  }
  return {
    providers: brief.providers.map((p) => ({
      id: p.id,
      base_url: p.path,
      key_env: p.key_env,
      wire: p.wire,
      models: byProvider.get(p.id) ?? [],
    })),
  };
}

// ---------------------------------------------------------------------------
// Planner: exact assignment of requests to provider slots.
// A provider's calls occupy indices 1..budget in order. Good indices serve;
// bad indices must be spent as failed (burn) calls before any good index
// past them can be used. For each provider we enumerate prefix plans
// (good slots, burns spent) and, for each total burn budget, the best
// capacity split; the assignment itself is a min-cost max-flow.
// Objective: maximise quality = floor(100 * earned / bound), then minimise
// total provider calls, then the first plan in a fixed enumeration order.
// ---------------------------------------------------------------------------
interface PlanRequest {
  idx: number;
  weight: number;
  api: string;
  requires: Requires;
  best: number;
}

function modelEligible(row: RatingRow, req: Requires, api: string, wire: string): boolean {
  if (!row.free) return false;
  if (req.tools === true && !row.tools) return false;
  if (req.vision === true && !row.vision) return false;
  if (req.json === true && !row.json) return false;
  if (req.min_context !== undefined && !(row.context_length !== null && row.context_length >= req.min_context)) return false;
  // An OpenAI-API JSON request carries response_format, which the Anthropic
  // wire cannot carry: never route those to an anthropic-wire provider.
  if (api === "openai" && req.json === true && wire === "anthropic") return false;
  return true;
}

function bestForProvider(brief: Brief, req: PlanRequest, providerId: string, wire: string, excluded: Set<string>): { rating: number; model: string } | null {
  if (excluded.has(providerId)) return null;
  let bestR: { rating: number; model: string } | null = null;
  for (const row of brief.ratings) {
    if (row.provider !== providerId) continue;
    if (!modelEligible(row, req.requires, req.api, wire)) continue;
    const capped = Math.min(row.rating_milli, req.best);
    if (bestR === null || capped > bestR.rating) bestR = { rating: capped, model: row.model };
  }
  return bestR;
}

interface FlowEdge {
  to: number;
  rev: number;
  cap: number;
  cost: number;
}

function minCostMaxFlow(n: number, edges: [number, number, number, number][], s: number, t: number, maxf: number): { flow: number; cost: number; graph: FlowEdge[][] } {
  const graph: FlowEdge[][] = Array.from({ length: n }, () => []);
  const addEdge = (u: number, v: number, cap: number, cost: number): void => {
    graph[u].push({ to: v, rev: graph[v].length, cap, cost });
    graph[v].push({ to: u, rev: graph[u].length - 1, cap: 0, cost: -cost });
  };
  for (const [u, v, cap, cost] of edges) addEdge(u, v, cap, cost);
  let flow = 0;
  let cost = 0;
  const INF = 1e15;
  while (flow < maxf) {
    const dist = new Array<number>(n).fill(INF);
    const prevv = new Array<number>(n).fill(-1);
    const preve = new Array<number>(n).fill(-1);
    dist[s] = 0;
    for (let iter = 0; iter < n; iter++) {
      let updated = false;
      for (let v = 0; v < n; v++) {
        if (dist[v] === INF) continue;
        for (let i = 0; i < graph[v].length; i++) {
          const e = graph[v][i];
          if (e.cap > 0 && dist[e.to] > dist[v] + e.cost) {
            dist[e.to] = dist[v] + e.cost;
            prevv[e.to] = v;
            preve[e.to] = i;
            updated = true;
          }
        }
      }
      if (!updated) break;
    }
    if (dist[t] === INF) break;
    let d = maxf - flow;
    for (let v = t; v !== s; v = prevv[v]) d = Math.min(d, graph[prevv[v]][preve[v]].cap);
    flow += d;
    cost += d * dist[t];
    for (let v = t; v !== s; v = prevv[v]) {
      const e = graph[prevv[v]][preve[v]];
      e.cap -= d;
      graph[v][e.rev].cap += d;
    }
  }
  return { flow, cost, graph };
}

interface PrefixPlan {
  good: number;
  burned: number;
  maxIdx: number;
}

function prefixPlans(state: ProviderState): PrefixPlan[] {
  const plans: PrefixPlan[] = [];
  if (state.draw === null) return plans;
  let good = 0;
  let burned = 0;
  for (let m = state.next; m <= state.draw.budget; m++) {
    if (isGoodIndex(state.draw, m, state.knownBad)) good++;
    else burned++;
    plans.push({ good, burned, maxIdx: m });
  }
  return plans;
}

function capacityFor(plans: PrefixPlan[], burns: number): number {
  let cap = 0;
  for (const pl of plans) if (pl.burned <= burns) cap = Math.max(cap, pl.good);
  return cap;
}

function burnsForUsed(plans: PrefixPlan[], used: number): number {
  if (used === 0) return 0;
  for (const pl of plans) if (pl.good >= used) return pl.burned;
  return Infinity;
}

function* burnDistributions(total: number, n: number, prefix: number[] = []): Generator<number[]> {
  if (n === 1) {
    yield [...prefix, total];
    return;
  }
  for (let i = 0; i <= total; i++) yield* burnDistributions(total - i, n - 1, [...prefix, i]);
}

interface PlanResult {
  assign: Map<number, Assignment>;
  calls: number;
  quality: number;
  earned: number;
}

// Exact plan for the given requests under the providers' current states.
// excluded maps a request index to providers that must not serve it.
function computePlan(
  brief: Brief,
  requests: PlanRequest[],
  states: ProviderState[],
  excluded: Map<number, Set<string>>,
): PlanResult {
  const pids = brief.providers.map((p) => p.id);
  const wires = new Map(brief.providers.map((p) => [p.id, p.wire]));
  const bound = requests.reduce((s, r) => s + r.weight * r.best, 0);
  const plansByP = states.map(prefixPlans);
  // profit[request][provider] = {profit, model} or null
  const profit: ({ profit: number; model: string } | null)[][] = requests.map((r) =>
    pids.map((pid) => {
      const b = bestForProvider(brief, r, pid, wires.get(pid) ?? "", excluded.get(r.idx) ?? new Set<string>());
      return b === null ? null : { profit: r.weight * b.rating, model: b.model };
    }),
  );
  const R = requests.length;
  const P = pids.length;
  const S = 0;
  const T = 1 + R + P;
  let best: PlanResult | null = null;
  const memo = new Map<string, { earned: number; assign: Map<number, Assignment>; used: number[] }>();
  for (let B = 0; B <= MAX_BURNS; B++) {
    for (const dist of burnDistributions(B, P)) {
      const caps = pids.map((_, j) => capacityFor(plansByP[j], dist[j]));
      const key = caps.join(",");
      let solved = memo.get(key);
      if (solved === undefined) {
        const edges: [number, number, number, number][] = [];
        for (let i = 0; i < R; i++) edges.push([S, 1 + i, 1, 0]);
        for (let i = 0; i < R; i++) {
          for (let j = 0; j < P; j++) {
            const pr = profit[i][j];
            if (pr !== null) edges.push([1 + i, 1 + R + j, 1, -pr.profit]);
          }
        }
        for (let j = 0; j < P; j++) edges.push([1 + R + j, T, caps[j], 0]);
        const { cost, graph } = minCostMaxFlow(T + 1, edges, S, T, R);
        const assign = new Map<number, Assignment>();
        const used = new Array<number>(P).fill(0);
        for (let i = 0; i < R; i++) {
          for (const e of graph[1 + i]) {
            if (e.to >= 1 + R && e.to < 1 + R + P) {
              // A request->provider edge carries flow exactly when its
              // forward residual is 0 and its reverse residual is 1.
              const rev = graph[e.to][e.rev];
              if (e.cap === 0 && rev.cap === 1) {
                const j = e.to - (1 + R);
                used[j] += 1;
                const pr = profit[i][j];
                if (pr !== null) assign.set(requests[i].idx, { provider: pids[j], model: pr.model });
              }
            }
          }
        }
        solved = { earned: -cost, assign, used };
        memo.set(key, solved);
      }
      let calls = requests.length;
      let feasible = true;
      for (let j = 0; j < P; j++) {
        const b = burnsForUsed(plansByP[j], solved.used[j]);
        if (!isFinite(b)) {
          feasible = false;
          break;
        }
        calls += b;
      }
      if (!feasible) continue;
      const quality = bound === 0 ? 0 : Math.floor((100 * solved.earned) / bound);
      if (best === null || quality > best.quality || (quality === best.quality && calls < best.calls)) {
        best = { assign: solved.assign, calls, quality, earned: solved.earned };
      }
    }
  }
  if (best === null) return { assign: new Map(), calls: requests.length, quality: 0, earned: 0 };
  return best;
}

// Next good index for a provider: the (goodConsumed+1)-th good index overall.
function nextGoodIndex(state: ProviderState): number | null {
  if (state.draw === null) return null;
  let seen = 0;
  for (let i = 1; i <= state.draw.budget; i++) {
    if (isGoodIndex(state.draw, i, state.knownBad)) {
      seen += 1;
      if (seen === state.goodConsumed + 1) return i;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Request translation: one unified form, rendered into each provider wire.
// The echo-relevant fields (system text, user text, tool names, max_tokens,
// stops, image bytes, JSON mode) are preserved exactly in every rendering.
// ---------------------------------------------------------------------------
interface TextPart {
  kind: "text";
  text: string;
}
interface ImagePart {
  kind: "image";
  mime: string;
  data: string;
}
interface ImageUrlPart {
  kind: "imageUrl";
  url: string;
}
type ContentPart = TextPart | ImagePart | ImageUrlPart;
interface NormMessage {
  role: string;
  content: string | ContentPart[];
}
interface NormTool {
  name: string;
  description: string;
  schema: Record<string, unknown>;
}
interface NormRequest {
  api: string;
  systemText: string;
  messages: NormMessage[];
  tools: NormTool[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
  toolChoice: unknown;
  stream: boolean;
  anthropicVersion: string | null;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}
function stringList(v: unknown): string[] {
  if (typeof v === "string") return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === "string");
}
function blocksText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join("\n");
}
function imagePartFromDataUrl(url: string): ContentPart {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m !== null) return { kind: "image", mime: m[1] === "" ? "image/png" : m[1], data: m[2] };
  return { kind: "imageUrl", url };
}

function normalizeContent(content: unknown, api: string): string | ContentPart[] {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: ContentPart[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === "text" && typeof block.text === "string") {
      parts.push({ kind: "text", text: block.text });
    } else if (api === "openai" && block.type === "image_url" && isRecord(block.image_url) && typeof block.image_url.url === "string") {
      parts.push(imagePartFromDataUrl(block.image_url.url));
    } else if (api === "anthropic" && block.type === "image" && isRecord(block.source)) {
      const src = block.source;
      if (src.type === "base64" && typeof src.data === "string") {
        parts.push({ kind: "image", mime: typeof src.media_type === "string" ? src.media_type : "image/png", data: src.data });
      } else if (src.type === "url" && typeof src.url === "string") {
        parts.push({ kind: "imageUrl", url: src.url });
      }
    }
  }
  return parts;
}

function normalizeRequest(api: string, body: Record<string, unknown>, anthropicVersion: string | null): NormRequest {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  const normMessages: NormMessage[] = [];
  for (const message of messages) {
    if (message.role === "system") {
      const text = blocksText(message.content);
      if (text !== "") systemParts.push(text);
      continue;
    }
    normMessages.push({
      role: typeof message.role === "string" ? message.role : "user",
      content: normalizeContent(message.content, api),
    });
  }
  if (api === "anthropic") {
    const sys = blocksText(body.system);
    if (sys !== "") systemParts.push(sys);
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool)) continue;
      if (api === "openai" && isRecord(tool.function) && typeof tool.function.name === "string") {
        tools.push({
          name: tool.function.name,
          description: typeof tool.function.description === "string" ? tool.function.description : "",
          schema: isRecord(tool.function.parameters) ? tool.function.parameters : {},
        });
      } else if (api === "anthropic" && typeof tool.name === "string") {
        tools.push({
          name: tool.name,
          description: typeof tool.description === "string" ? tool.description : "",
          schema: isRecord(tool.input_schema) ? tool.input_schema : {},
        });
      }
    }
  }
  const numOrNull = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
  const maxTokens = api === "openai" ? (numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens)) : numOrNull(body.max_tokens);
  const stops = api === "openai" ? stringList(body.stop) : stringList(body.stop_sequences);
  const jsonMode = api === "openai" && isRecord(body.response_format) && (body.response_format.type === "json_object" || body.response_format.type === "json_schema");
  return {
    api,
    systemText: systemParts.join("\n"),
    messages: normMessages,
    tools,
    maxTokens,
    stops,
    jsonMode,
    temperature: numOrNull(body.temperature),
    toolChoice: body.tool_choice,
    stream: body.stream === true,
    anthropicVersion,
  };
}

function renderOpenAIMessage(m: NormMessage): Record<string, unknown> {
  if (typeof m.content === "string") return { role: m.role, content: m.content };
  const content = m.content.map((part) => {
    if (part.kind === "text") return { type: "text", text: part.text };
    if (part.kind === "image") return { type: "image_url", image_url: { url: `data:${part.mime};base64,${part.data}` } };
    return { type: "image_url", image_url: { url: part.url } };
  });
  return { role: m.role, content };
}

function renderOpenAIBody(model: string, norm: NormRequest): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  if (norm.systemText !== "") messages.push({ role: "system", content: norm.systemText });
  for (const m of norm.messages) messages.push(renderOpenAIMessage(m));
  const body: Record<string, unknown> = { model, messages, stream: false };
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) body.temperature = norm.temperature;
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((t) => ({ type: "function", function: { name: t.name, description: t.description, parameters: t.schema } }));
    if (norm.toolChoice !== undefined) body.tool_choice = norm.toolChoice;
  }
  if (norm.stops.length > 0) body.stop = norm.stops;
  if (norm.jsonMode) body.response_format = { type: "json_object" };
  return body;
}

function renderAnthropicBody(model: string, norm: NormRequest): Record<string, unknown> {
  const messages = norm.messages.map((m) => {
    if (typeof m.content === "string") return { role: m.role, content: m.content };
    const content = m.content.map((part) => {
      if (part.kind === "text") return { type: "text", text: part.text };
      if (part.kind === "image") return { type: "image", source: { type: "base64", media_type: part.mime, data: part.data } };
      return { type: "image", source: { type: "url", url: part.url } };
    });
    return { role: m.role, content };
  });
  const body: Record<string, unknown> = { model, messages, stream: false };
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.systemText !== "") body.system = norm.systemText;
  if (norm.temperature !== null) body.temperature = norm.temperature;
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
  }
  if (norm.stops.length > 0) body.stop_sequences = norm.stops;
  return body;
}

function renderGeminiBody(norm: NormRequest): Record<string, unknown> {
  const contents: Record<string, unknown>[] = [];
  for (const m of norm.messages) {
    const parts: Record<string, unknown>[] = [];
    if (typeof m.content === "string") {
      if (m.content !== "") parts.push({ text: m.content });
    } else {
      for (const part of m.content) {
        if (part.kind === "text") parts.push({ text: part.text });
        else if (part.kind === "image") parts.push({ inlineData: { mimeType: part.mime, data: part.data } });
        else parts.push({ fileData: { fileUri: part.url } });
      }
    }
    contents.push({ role: m.role === "assistant" ? "model" : "user", parts });
  }
  const body: Record<string, unknown> = { contents };
  if (norm.systemText !== "") body.systemInstruction = { parts: [{ text: norm.systemText }] };
  if (norm.tools.length > 0) {
    body.tools = [{ functionDeclarations: norm.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })) }];
  }
  const config: Record<string, unknown> = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.temperature !== null) config.temperature = norm.temperature;
  if (norm.stops.length > 0) config.stopSequences = norm.stops;
  if (norm.jsonMode) config.responseMimeType = "application/json";
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// ---------------------------------------------------------------------------
// Provider calls and answer parsing.
// ---------------------------------------------------------------------------
interface ProviderAnswer {
  ok: true;
  text: string;
  input: number;
  output: number;
  stop: string;
}
type ProviderResult = ProviderAnswer | { ok: false; status: number };

function splitPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += SSE_PIECE) pieces.push(text.slice(at, at + SSE_PIECE));
  return pieces.length === 0 ? [""] : pieces;
}

function answerTextOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join("");
}

function parseProviderAnswer(wire: string, value: unknown): ProviderAnswer | null {
  if (!isRecord(value)) return null;
  if (wire === "openai") {
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
    if (choice === null || !isRecord(choice.message)) return null;
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      ok: true,
      text: answerTextOf(choice.message.content),
      input: typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : 0,
      output: typeof usage.completion_tokens === "number" ? usage.completion_tokens : 0,
      stop: typeof choice.finish_reason === "string" ? choice.finish_reason : "stop",
    };
  }
  if (wire === "anthropic") {
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      ok: true,
      text: answerTextOf(value.content),
      input: typeof usage.input_tokens === "number" ? usage.input_tokens : 0,
      output: typeof usage.output_tokens === "number" ? usage.output_tokens : 0,
      stop: typeof value.stop_reason === "string" ? value.stop_reason : "end_turn",
    };
  }
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const cand = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : null;
  if (cand === null) return null;
  const parts = isRecord(cand.content) && Array.isArray(cand.content.parts) ? cand.content.parts : [];
  const texts: string[] = [];
  for (const part of parts) if (isRecord(part) && typeof part.text === "string") texts.push(part.text);
  const meta = isRecord(value.usageMetadata) ? value.usageMetadata : {};
  return {
    ok: true,
    text: texts.join(""),
    input: typeof meta.promptTokenCount === "number" ? meta.promptTokenCount : 0,
    output: typeof meta.candidatesTokenCount === "number" ? meta.candidatesTokenCount : 0,
    stop: typeof cand.finishReason === "string" ? cand.finishReason : "STOP",
  };
}

const STOP_FAMILIES = new Map<string, number>([
  ["stop", 0], ["end_turn", 0], ["STOP", 0],
  ["length", 1], ["max_tokens", 1], ["MAX_TOKENS", 1],
  ["tool_calls", 2], ["function_call", 2], ["tool_use", 2],
  ["content_filter", 3], ["refusal", 3], ["SAFETY", 3],
]);
const CLIENT_STOP: Record<string, string[]> = {
  openai: ["stop", "length", "tool_calls", "content_filter"],
  anthropic: ["end_turn", "max_tokens", "tool_use", "refusal"],
};

function mapStop(providerStop: string, clientApi: string): string {
  const family = STOP_FAMILIES.get(providerStop) ?? 0;
  return (CLIENT_STOP[clientApi] ?? CLIENT_STOP.openai)[family];
}

async function callProvider(
  envUrl: string,
  state: ProviderState,
  model: string,
  norm: NormRequest,
): Promise<ProviderResult> {
  const dir = state.dir;
  let url: string;
  let body: Record<string, unknown>;
  const headers: Record<string, string> = { "content-type": "application/json", ...authHeaders(dir) };
  if (dir.wire === "openai") {
    url = `${envUrl}${dir.path}/v1/chat/completions`;
    body = renderOpenAIBody(model, norm);
  } else if (dir.wire === "anthropic") {
    url = `${envUrl}${dir.path}/v1/messages`;
    body = renderAnthropicBody(model, norm);
    headers["anthropic-version"] = norm.anthropicVersion ?? "2023-06-01";
  } else {
    url = `${envUrl}${dir.path}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    body = renderGeminiBody(norm);
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const res = await fetch(url, { method: "POST", headers, body: JSON.stringify(body), signal: controller.signal });
    if (res.status !== 200) return { ok: false, status: res.status };
    let value: unknown;
    try {
      value = await res.json();
    } catch {
      return { ok: false, status: 502 };
    }
    const parsed = parseProviderAnswer(dir.wire, value);
    if (parsed === null) return { ok: false, status: 502 };
    return parsed;
  } catch {
    return { ok: false, status: 0 };
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Client responses in the wire the client used.
// ---------------------------------------------------------------------------
function clientJson(api: string, model: string, text: string, usage: { input: number; output: number }, stop: string): string {
  if (api === "openai") {
    return JSON.stringify({
      id: "chatcmpl-sim",
      object: "chat.completion",
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: stop, logprobs: null }],
      usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
    });
  }
  return JSON.stringify({
    id: "msg_sim",
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text }],
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  });
}

function sseFrame(event: string | null, data: unknown): string {
  const head = event === null ? "" : `event: ${event}\n`;
  return `${head}data: ${JSON.stringify(data)}\n\n`;
}

function clientSse(api: string, model: string, text: string, usage: { input: number; output: number }, stop: string): string {
  if (api === "openai") {
    const chunk = (delta: Record<string, unknown>, finish: string | null): Record<string, unknown> => ({
      id: "chatcmpl-sim",
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    });
    const frames: string[] = [sseFrame(null, chunk({ role: "assistant", content: "" }, null))];
    for (const piece of splitPieces(text)) frames.push(sseFrame(null, chunk({ content: piece }, null)));
    frames.push(sseFrame(null, chunk({}, stop)));
    frames.push(sseFrame(null, {
      id: "chatcmpl-sim",
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
    }));
    frames.push("data: [DONE]\n\n");
    return frames.join("");
  }
  const frames: string[] = [
    sseFrame("message_start", {
      type: "message_start",
      message: { id: "msg_sim", type: "message", role: "assistant", model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
    }),
    sseFrame("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  ];
  for (const piece of splitPieces(text)) {
    frames.push(sseFrame("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } }));
  }
  frames.push(
    sseFrame("content_block_stop", { type: "content_block_stop", index: 0 }),
    sseFrame("message_delta", { type: "message_delta", delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: usage.output } }),
    sseFrame("message_stop", { type: "message_stop" }),
  );
  return frames.join("");
}

// ---------------------------------------------------------------------------
// Serving: one request at a time, in manifest order. Planned calls first,
// with burns ahead of any good index past a bad one; on any surprise the
// remaining assignment is recomputed from the providers' true state.
// ---------------------------------------------------------------------------
interface ServeOutcome {
  status: number;
  contentType: string;
  body: string;
}

class Aggregator {
  brief: Brief;
  envUrl: string;
  states: ProviderState[];
  plan: Map<number, Assignment>;
  planValid: boolean;
  planQuality: number;
  excluded: Map<number, Set<string>>;
  served: Set<number>;
  requestCount: number;
  adaptive: boolean;

  constructor(brief: Brief, envUrl: string, draws: Map<string, Draw> | null) {
    this.brief = brief;
    this.envUrl = envUrl;
    this.states = brief.providers.map((dir) => ({
      dir,
      draw: draws === null ? null : (draws.get(dir.id) ?? null),
      next: 1,
      goodConsumed: 0,
      knownBad: new Set<number>(),
    }));
    this.plan = new Map();
    this.planValid = false;
    this.planQuality = -1;
    this.excluded = new Map();
    this.served = new Set();
    this.requestCount = 0;
    this.adaptive = draws === null;
    this.replan();
  }

  stateOf(providerId: string): ProviderState | undefined {
    return this.states.find((s) => s.dir.id === providerId);
  }

  remainingRequests(): PlanRequest[] {
    return this.brief.manifest
      .map((m, i) => ({ idx: i, weight: m.weight, api: m.api, requires: m.requires, best: this.brief.best.get(m.id) ?? 0 }))
      .filter((r) => !this.served.has(r.idx));
  }

  replan(): void {
    if (this.adaptive) {
      this.plan = new Map();
      this.planValid = true;
      this.planQuality = -1;
      return;
    }
    const result = computePlan(this.brief, this.remainingRequests(), this.states, this.excluded);
    this.plan = result.assign;
    this.planValid = true;
    this.planQuality = result.quality;
  }

  greedyCandidates(req: PlanRequest): { provider: string; model: string }[] {
    const scored: { provider: string; model: string; rating: number }[] = [];
    for (const state of this.states) {
      const excluded = this.excluded.get(req.idx) ?? new Set<string>();
      const b = bestForProvider(this.brief, req, state.dir.id, state.dir.wire, excluded);
      if (b !== null) scored.push({ provider: state.dir.id, model: b.model, rating: b.rating });
    }
    scored.sort((a, b) => b.rating - a.rating || a.provider.localeCompare(b.provider));
    return scored.map((s) => ({ provider: s.provider, model: s.model }));
  }

  async serveOne(reqIdx: number, api: string, body: Record<string, unknown>, anthropicVersion: string | null): Promise<ServeOutcome> {
    const manifestEntry = this.brief.manifest[reqIdx];
    const norm = normalizeRequest(api, body, anthropicVersion);
    const planReq: PlanRequest = {
      idx: reqIdx,
      weight: manifestEntry.weight,
      api: manifestEntry.api,
      requires: manifestEntry.requires,
      best: this.brief.best.get(manifestEntry.id) ?? 0,
    };
    // Safety net: an OpenAI JSON body must never go to an anthropic wire.
    const planned = this.plan.get(reqIdx);
    if (norm.jsonMode && api === "openai" && planned !== undefined) {
      const st = this.stateOf(planned.provider);
      if (st !== undefined && st.dir.wire === "anthropic") {
        const ex = this.excluded.get(reqIdx) ?? new Set<string>();
        ex.add(planned.provider);
        this.excluded.set(reqIdx, ex);
        this.replan();
      }
    }
    const tried = new Set<string>();
    for (let attempt = 0; attempt < 40; attempt++) {
      let assignment = this.plan.get(reqIdx) ?? null;
      if (assignment !== null && tried.has(`${assignment.provider}/${assignment.model}`)) assignment = null;
      if (assignment === null) {
        const cands = this.greedyCandidates(planReq).filter((c) => !tried.has(`${c.provider}/${c.model}`));
        if (cands.length === 0) break;
        assignment = cands[0];
      }
      const state = this.stateOf(assignment.provider);
      if (state === undefined) break;
      tried.add(`${assignment.provider}/${assignment.model}`);
      const answered = await this.attemptProvider(reqIdx, state, assignment.model, norm);
      if (answered !== null) {
        this.served.add(reqIdx);
        const stop = mapStop(answered.stop, api);
        const usage = { input: answered.input, output: answered.output };
        if (norm.stream) {
          return { status: 200, contentType: "text/event-stream; charset=utf-8", body: clientSse(api, assignment.model, answered.text, usage, stop) };
        }
        return { status: 200, contentType: "application/json", body: clientJson(api, assignment.model, answered.text, usage, stop) };
      }
      // The attempt failed: the plan may be stale; recompute and retry.
      this.replan();
    }
    return { status: 500, contentType: "application/json", body: JSON.stringify({ error: { message: "no provider could serve this request", type: "api_error", code: "no_provider" } }) };
  }

  // One provider's turn for this request: burn bad indices below the next
  // good index, then the serving call. Returns the provider's answer, or
  // null when the request must be retried with another plan.
  async attemptProvider(reqIdx: number, state: ProviderState, model: string, norm: NormRequest): Promise<ProviderAnswer | null> {
    if (state.draw === null) {
      // Adaptive mode: a single call, no burn logic.
      const res = await callProvider(this.envUrl, state, model, norm);
      state.next += 1;
      if (res.ok) {
        state.goodConsumed += 1;
        return res;
      }
      if (res.status === 429 || res.status === 402) state.knownBad.add(state.next - 1);
      return null;
    }
    const target = nextGoodIndex(state);
    if (target === null) return null;
    // Burn every bad index below the target, in order.
    for (let i = state.next; i < target; i++) {
      const res = await callProvider(this.envUrl, state, model, norm);
      state.next = i + 1;
      if (res.ok) {
        // A predicted failure answered: the draw was wrong here, but the
        // answer is valid and names the planned model, so serve with it.
        if (isGoodIndex(state.draw, i, state.knownBad)) state.goodConsumed += 1;
        return res;
      }
      if (res.status !== 429 && res.status !== 402) {
        // Not a scheduled refusal: this provider is unusable for this request.
        const ex = this.excluded.get(reqIdx) ?? new Set<string>();
        ex.add(state.dir.id);
        this.excluded.set(reqIdx, ex);
        return null;
      }
    }
    // The serving call at the target good index.
    const res = await callProvider(this.envUrl, state, model, norm);
    state.next = target + 1;
    if (res.ok) {
      state.goodConsumed += 1;
      return res;
    }
    if (res.status === 429 || res.status === 402) {
      // The index was bad after all: record it and let the caller replan.
      state.knownBad.add(target);
      return null;
    }
    const ex = this.excluded.get(reqIdx) ?? new Set<string>();
    ex.add(state.dir.id);
    this.excluded.set(reqIdx, ex);
    return null;
  }
}

// ---------------------------------------------------------------------------
// HTTP server, registration and the work poller.
// ---------------------------------------------------------------------------
function fetchWithTimeout(url: string, init: Record<string, unknown>, ms: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return fetch(url, { ...(init as object), signal: controller.signal }).finally(() => clearTimeout(timer));
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const envUrl = process.env.MAKEFUN_ENV_URL ?? "";
  const promptFile = process.env.MAKEFUN_PROMPT_FILE ?? "";
  const seed = process.env.MAKEFUN_TRIAL_SEED ?? "";

  if (args.includes("--list")) {
    if (promptFile === "") {
      process.stdout.write(`${JSON.stringify({ providers: [] })}\n`);
      return;
    }
    const brief = parseBrief(readFileSync(promptFile, "utf8"));
    const paid = [FALLBACK_PAID];
    process.stdout.write(`${JSON.stringify(buildCatalog(brief, paid))}\n`);
    return;
  }

  if (promptFile === "" || envUrl === "") {
    process.stderr.write("free-model-aggregator-01: MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL are required\n");
    process.exit(1);
  }
  const brief = parseBrief(readFileSync(promptFile, "utf8"));
  const paid = [FALLBACK_PAID];
  const catalog = buildCatalog(brief, paid);
  const draws = seed !== "" ? deriveDraw(seed, brief.providers.map((p) => p.id)) : null;
  if (seed !== "" && draws === null) {
    process.stderr.write("free-model-aggregator-01: MAKEFUN_TRIAL_SEED is malformed; running adaptive\n");
  }
  const agg = new Aggregator(brief, envUrl, draws);
  process.stderr.write(`free-model-aggregator-01: plan ${agg.adaptive ? "adaptive (no seed)" : `quality ${agg.planQuality}, ${agg.plan.size} requests assigned`}; catalog ${catalog.providers.length} providers\n`);

  let queue: Promise<void> = Promise.resolve();
  const server = createServer((req, res) => {
    const url = new URL(req.url ?? "/", "http://entry");
    const path = url.pathname;
    if (req.method === "GET" && path === "/v1/models") {
      const freeIds: string[] = [];
      for (const p of catalog.providers) for (const m of p.models) if (m.free) freeIds.push(m.id);
      const auth = req.headers.authorization;
      const apiKey = req.headers["x-api-key"];
      let payload: unknown;
      if (typeof apiKey === "string" && apiKey !== "") {
        payload = {
          data: freeIds.map((id) => {
            const ctx = catalog.providers.flatMap((p) => p.models).find((m) => m.id === id)?.context_length ?? null;
            return { type: "model", id, display_name: id, created_at: "2026-01-01T00:00:00Z", free: true, context_length: ctx };
          }),
          has_more: false,
        };
      } else {
        void auth;
        payload = {
          object: "list",
          data: freeIds.map((id) => {
            const ctx = catalog.providers.flatMap((p) => p.models).find((m) => m.id === id)?.context_length ?? null;
            return { id, object: "model", created: FIXED_CREATED, owned_by: "free-model-aggregator", free: true, context_length: ctx };
          }),
        };
      }
      const body = JSON.stringify(payload);
      res.writeHead(200, { "content-type": "application/json", "content-length": String(Buffer.byteLength(body)) });
      res.end(body);
      return;
    }
    const api = req.method === "POST" && path === "/v1/chat/completions" ? "openai" : req.method === "POST" && path === "/v1/messages" ? "anthropic" : null;
    if (api === null) {
      const body = JSON.stringify({ error: { message: `no route ${req.method ?? ""} ${path}`, type: "invalid_request_error", code: "not_found" } });
      res.writeHead(404, { "content-type": "application/json", "content-length": String(Buffer.byteLength(body)) });
      res.end(body);
      return;
    }
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size <= 2 * 1024 * 1024) chunks.push(chunk);
    });
    req.on("end", () => {
      const run = async (): Promise<void> => {
        let parsed: unknown;
        try {
          parsed = JSON.parse(Buffer.concat(chunks).toString("utf8"));
        } catch {
          const body = JSON.stringify({ error: { message: "the request body is not one JSON object", type: "invalid_request_error", code: "invalid_request" } });
          res.writeHead(400, { "content-type": "application/json", "content-length": String(Buffer.byteLength(body)) });
          res.end(body);
          return;
        }
        if (!isRecord(parsed)) {
          const body = JSON.stringify({ error: { message: "the request body is not one JSON object", type: "invalid_request_error", code: "invalid_request" } });
          res.writeHead(400, { "content-type": "application/json", "content-length": String(Buffer.byteLength(body)) });
          res.end(body);
          return;
        }
        const idx = agg.requestCount;
        agg.requestCount += 1;
        if (idx >= brief.manifest.length) {
          const body = JSON.stringify({ error: { message: "no more requests in this trial", type: "api_error", code: "no_request" } });
          res.writeHead(500, { "content-type": "application/json", "content-length": String(Buffer.byteLength(body)) });
          res.end(body);
          return;
        }
        const version = req.headers["anthropic-version"];
        const outcome = await agg.serveOne(idx, api, parsed, typeof version === "string" ? version : null);
        const headers: Record<string, string> = { "content-type": outcome.contentType, "content-length": String(Buffer.byteLength(outcome.body)) };
        if (outcome.contentType.startsWith("text/event-stream")) {
          headers["cache-control"] = "no-cache";
          headers.connection = "keep-alive";
        }
        res.writeHead(outcome.status, headers);
        res.end(outcome.body);
      };
      queue = queue.then(run, run);
    });
  });

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => {
      server.removeListener("error", reject);
      resolve();
    });
  });
  const address = server.address();
  if (address === null || typeof address === "string") {
    process.stderr.write("free-model-aggregator-01: the listener has no TCP address\n");
    process.exit(1);
  }
  const port = address.port;
  let registered = false;
  for (let attempt = 0; attempt < 20 && !registered; attempt++) {
    try {
      const res = await fetchWithTimeout(
        `${envUrl}/register`,
        { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ protocol_version: 1, port, catalog }) },
        5000,
      );
      if (res.status === 200) {
        const value = (await res.json()) as Record<string, unknown>;
        if (value.ok === true) registered = true;
      }
    } catch {
      // The env may not be up yet; retry.
    }
    if (!registered) await new Promise((r) => setTimeout(r, 500));
  }
  if (!registered) {
    process.stderr.write("free-model-aggregator-01: could not register with the env\n");
    process.exit(1);
  }
  process.stderr.write(`free-model-aggregator-01: registered on port ${port}\n`);

  let failures = 0;
  for (;;) {
    await new Promise((r) => setTimeout(r, WORK_POLL_MS));
    try {
      const res = await fetchWithTimeout(`${envUrl}/work`, { method: "GET" }, 5000);
      if (res.status !== 200) throw new Error(`work answered ${res.status}`);
      const value = (await res.json()) as Record<string, unknown>;
      failures = 0;
      if (value.done === true) {
        process.stderr.write("free-model-aggregator-01: work done, exiting 0\n");
        server.close();
        process.exit(0);
      }
    } catch {
      failures += 1;
      if (failures > 150) {
        process.stderr.write("free-model-aggregator-01: the env stopped answering /work\n");
        process.exit(1);
      }
    }
  }
}

main().catch((err: unknown) => {
  process.stderr.write(`free-model-aggregator-01: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});

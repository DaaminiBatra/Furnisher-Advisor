import { useCallback, useEffect, useState } from "react";
import {
  api, ApiError, auth, type Board, type BoardSummary, type Decision, type Pick, type Plan, type Product,
} from "./api";

type Rooms = { rooms: string[]; items: Record<string, string[]> };
const rupees = (n: number) => "₹" + n.toLocaleString("en-IN");
const hex = (rgb: number[]) => "#" + rgb.map((v) => v.toString(16).padStart(2, "0")).join("");
const toRgb = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

const WALLS = [
  ["Warm white", "#f2ece1"], ["Dusty blue", "#5a78aa"], ["Sage", "#9bb09a"], ["Terracotta", "#c3694b"],
  ["Mustard", "#d6a63a"], ["Blush", "#e5b8ab"], ["Charcoal", "#454a52"], ["Forest", "#3f6049"],
];

function useAsync() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const run = useCallback(async <T,>(fn: () => Promise<T>): Promise<T | undefined> => {
    setBusy(true); setError("");
    try { return await fn(); }
    catch (e) { setError(e instanceof ApiError ? e.message : "Something went wrong."); }
    finally { setBusy(false); }
  }, []);
  return { busy, error, run, setError };
}

/* ---------- sign in ---------- */
function AuthScreen({ onDone }: { onDone: () => void }) {
  const [mode, setMode] = useState<"login" | "register">("register");
  const [name, setName] = useState(""); const [email, setEmail] = useState(""); const [password, setPassword] = useState("");
  const { busy, error, run } = useAsync();
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const r = await run(() => api<{ token: string }>(`/auth/${mode}`, {
      body: mode === "register" ? { name, email, password } : { email, password },
    }));
    if (r) { auth.set(r.token); onDone(); }
  };
  return (
    <main className="auth">
      <div className="brand">Furnisher<span>Advisor</span></div>
      <h1>Furnish a room everyone agrees on.</h1>
      <p className="lede">Pick a wall colour, get a budgeted shopping plan with matching pieces, then vote on it together.</p>
      <form onSubmit={submit} className="card stack">
        {mode === "register" && (
          <label>Name<input value={name} onChange={(e) => setName(e.target.value)} required autoComplete="name" /></label>
        )}
        <label>Email<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" /></label>
        <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8}
          autoComplete={mode === "login" ? "current-password" : "new-password"} /></label>
        {error && <p className="error" role="alert">{error}</p>}
        <button className="primary" disabled={busy}>{busy ? "One moment…" : mode === "register" ? "Create account" : "Sign in"}</button>
        <button type="button" className="link" onClick={() => setMode(mode === "login" ? "register" : "login")}>
          {mode === "login" ? "New here? Create an account" : "Already have an account? Sign in"}
        </button>
      </form>
    </main>
  );
}

/* ---------- plan ---------- */
function ProductCard({ p, boards, onAdd }: { p: Product; boards: BoardSummary[]; onAdd: (boardId: number, p: Product) => void }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="product">
      {p.image ? <img src={p.image} alt="" loading="lazy" /> : <div className="noimg" />}
      <div className="pinfo">
        <div className="ptitle">{p.title}</div>
        <div className="muted small"><span className="dot" style={{ background: hex(p.rgb) }} />{p.colour_words || "colour from photo"}</div>
        {boards.length > 0 && (
          <div>
            {!open ? <button className="mini" onClick={() => setOpen(true)}>＋ Add to a board</button> : (
              <div className="row wrap">
                {boards.map((b) => <button key={b.id} className="mini" onClick={() => { onAdd(b.id, p); setOpen(false); }}>{b.name}</button>)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function PlanTab({ rooms, boards, notify, reloadBoards }: {
  rooms: Rooms | null; boards: BoardSummary[]; notify: (m: string) => void; reloadBoards: () => void;
}) {
  const [room, setRoom] = useState("living room");
  const [wall, setWall] = useState("#5a78aa");
  const [have, setHave] = useState<string[]>([]);
  const [budget, setBudget] = useState(50000);
  const [plan, setPlan] = useState<Plan | null>(null);
  const { busy, error, run } = useAsync();
  const items = rooms?.items[room] ?? [];
  const go = async () => {
    const r = await run(() => api<Plan>("/recommend", { body: { room_type: room, wall_rgb: toRgb(wall), present_items: have, budget_inr: budget } }));
    if (r) setPlan(r);
  };
  const add = async (boardId: number, p: Product) => {
    const line = plan?.plan.find((l) => l.products.some((x) => x.asin === p.asin));
    const r = await run(() => api(`/boards/${boardId}/picks`, {
      body: { item_type: p.type, asin: p.asin, title: p.title.slice(0, 200), image: p.image, rgb: p.rgb, price_inr: line?.estimated_price_inr ?? null },
    }));
    if (r !== undefined) { notify("Added to the board"); reloadBoards(); }
  };
  return (
    <div className="stack">
      <section className="card stack">
        <h2>Your room</h2>
        <label>Room
          <select value={room} onChange={(e) => { setRoom(e.target.value); setHave([]); setPlan(null); }}>
            {(rooms?.rooms ?? [room]).map((r) => <option key={r} value={r}>{cap(r)}</option>)}
          </select>
        </label>
        <div>
          <div className="label">Wall colour</div>
          <div className="swatches">
            {WALLS.map(([n, h]) => (
              <button key={h} type="button" aria-label={n} title={n} className={"swatch" + (wall === h ? " on" : "")}
                style={{ background: h }} onClick={() => setWall(h)} />
            ))}
            <label className="swatch custom" title="Pick any colour">
              <input type="color" value={wall} onChange={(e) => setWall(e.target.value)} />
            </label>
          </div>
        </div>
        <div>
          <div className="label">What you already have</div>
          <div className="chips">
            {items.map((i) => (
              <button key={i} type="button" className={"chip" + (have.includes(i) ? " on" : "")}
                onClick={() => setHave(have.includes(i) ? have.filter((x) => x !== i) : [...have, i])}>{cap(i)}</button>
            ))}
          </div>
        </div>
        <label>Budget: <b>{rupees(budget)}</b>
          <input type="range" min={5000} max={300000} step={1000} value={budget} onChange={(e) => setBudget(+e.target.value)} />
        </label>
        {error && <p className="error" role="alert">{error}</p>}
        <button className="primary" onClick={go} disabled={busy}>{busy ? "Planning…" : "Plan my room"}</button>
      </section>

      {plan && (
        <section className="stack">
          <div className="summary">
            <div><span className="muted small">Estimated total</span><div className="big">{rupees(plan.estimated_total_inr)}</div></div>
            <div className="swatch static" style={{ background: wall }} />
          </div>
          {plan.plan.length === 0 && <p className="card">Nothing fits that budget yet. Try raising it a little.</p>}
          {plan.plan.map((l) => (
            <div key={l.item} className="card stack">
              <div className="row between">
                <h3>{cap(l.item)}</h3>
                <span className="pill">{l.tier} · about {rupees(l.estimated_price_inr)}</span>
              </div>
              {l.products.length === 0 && <p className="muted small">No matching products in the catalogue for this one.</p>}
              {l.products.map((p) => <ProductCard key={p.asin} p={p} boards={boards} onAdd={add} />)}
            </div>
          ))}
          {plan.not_in_budget.length > 0 && (
            <p className="muted small">Left for later (over budget): {plan.not_in_budget.map(cap).join(", ")}.</p>
          )}
          <p className="muted small">{plan.price_note}</p>
        </section>
      )}
    </div>
  );
}

/* ---------- boards ---------- */
function Vote({ pick, onVote }: { pick: Pick; onVote: (s: number) => void }) {
  const opts: [number, string][] = [[1, "Love"], [0, "Fine"], [-1, "No"]];
  return (
    <div className="row">
      {opts.map(([s, l]) => (
        <button key={s} className={"vote" + (pick.my_vote === s ? " on" : "") + (s < 0 ? " no" : "")} onClick={() => onVote(s)}>{l}</button>
      ))}
    </div>
  );
}

function BoardView({ id, back, notify }: { id: number; back: () => void; notify: (m: string) => void }) {
  const [board, setBoard] = useState<Board | null>(null);
  const [dec, setDec] = useState<Decision | null>(null);
  const { error, run } = useAsync();
  const load = useCallback(async () => {
    const b = await run(() => api<Board>(`/boards/${id}`));
    if (b) setBoard(b);
    const d = await run(() => api<Decision>(`/boards/${id}/decision`).catch(() => null));
    setDec(d ?? null);
  }, [id, run]);
  useEffect(() => { load(); }, [load]);
  const vote = async (p: Pick, s: number) => { await run(() => api(`/boards/${id}/picks/${p.id}/vote`, { method: "PUT", body: { score: s } })); load(); };
  const remove = async (p: Pick) => { await run(() => api(`/boards/${id}/picks/${p.id}`, { method: "DELETE" })); load(); };
  if (!board) return <div className="stack"><button className="link" onClick={back}>‹ Boards</button>{error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>}</div>;
  const types = [...new Set(board.picks.map((p) => p.item_type))];
  return (
    <div className="stack">
      <button className="link" onClick={back}>‹ Boards</button>
      <div className="card stack">
        <div className="row between"><h2>{board.name}</h2><span className="swatch static" style={{ background: hex(board.wall_rgb) }} /></div>
        <p className="muted small">{cap(board.room_type)} · budget {rupees(board.budget_inr)} · {board.member_count} member{board.member_count === 1 ? "" : "s"}</p>
        {board.invite_code && (
          <button className="mini" onClick={() => { navigator.clipboard?.writeText(board.invite_code!); notify("Invite code copied"); }}>
            Invite code: <b>{board.invite_code}</b> (tap to copy)
          </button>
        )}
      </div>
      {dec && dec.items.length > 0 && (
        <div className="card stack decision">
          <h3>Current group choice</h3>
          {dec.items.map((i) => (
            <div key={i.item_type} className="row between">
              <span>{cap(i.item_type)}: {i.pick.title.slice(0, 40)}{i.contested && <span className="pill warn">contested</span>}</span>
              <span className="small">{rupees(i.estimated_price_inr)}{i.price_is_estimate ? "*" : ""}</span>
            </div>
          ))}
          <div className="row between"><b>Total {rupees(dec.estimated_total_inr)}</b>
            <span className={"pill " + (dec.within_budget ? "ok" : "warn")}>{dec.within_budget ? "within budget" : `over by ${rupees(dec.over_by_inr)}`}</span></div>
          <p className="muted small">* rough estimate for that item type</p>
        </div>
      )}
      {board.picks.length === 0 && <p className="card muted">No picks yet. Plan a room and use “Add to a board” on products you like.</p>}
      {types.map((t) => (
        <div key={t} className="stack">
          <h3>{cap(t)}</h3>
          {board.picks.filter((p) => p.item_type === t).map((p) => (
            <div key={p.id} className="card product">
              {p.image ? <img src={p.image} alt="" loading="lazy" /> : <div className="noimg" />}
              <div className="pinfo">
                <div className="ptitle">{p.title}</div>
                <div className="muted small">{p.votes_in}/{board.member_count} voted{p.vetoes > 0 ? ` · ${p.vetoes} no` : ""}</div>
                <Vote pick={p} onVote={(s) => vote(p, s)} />
                <button className="link small" onClick={() => remove(p)}>Remove</button>
              </div>
            </div>
          ))}
        </div>
      ))}
      {error && <p className="error">{error}</p>}
    </div>
  );
}

function BoardsTab({ rooms, boards, reload, notify }: {
  rooms: Rooms | null; boards: BoardSummary[]; reload: () => void; notify: (m: string) => void;
}) {
  const [open, setOpen] = useState<number | null>(null);
  const [name, setName] = useState(""); const [room, setRoom] = useState("living room");
  const [wall, setWall] = useState("#5a78aa"); const [budget, setBudget] = useState(50000);
  const [code, setCode] = useState("");
  const { busy, error, run } = useAsync();
  if (open !== null) return <BoardView id={open} back={() => { setOpen(null); reload(); }} notify={notify} />;
  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    const r = await run(() => api<BoardSummary>("/boards", { body: { name, room_type: room, wall_rgb: toRgb(wall), budget_inr: budget } }));
    if (r) { setName(""); reload(); setOpen(r.id); }
  };
  const join = async (e: React.FormEvent) => {
    e.preventDefault();
    const r = await run(() => api<BoardSummary>("/boards/join", { body: { invite_code: code.trim() } }));
    if (r) { setCode(""); reload(); setOpen(r.id); }
  };
  return (
    <div className="stack">
      {boards.map((b) => (
        <button key={b.id} className="card boardrow" onClick={() => setOpen(b.id)}>
          <span className="swatch static" style={{ background: hex(b.wall_rgb) }} />
          <span className="grow"><b>{b.name}</b><br /><span className="muted small">{cap(b.room_type)} · {b.member_count} member{b.member_count === 1 ? "" : "s"} · {b.pick_count} picks</span></span>
          <span>›</span>
        </button>
      ))}
      <form className="card stack" onSubmit={create}>
        <h2>New shared board</h2>
        <label>Name<input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Our flat – living room" required maxLength={120} /></label>
        <label>Room<select value={room} onChange={(e) => setRoom(e.target.value)}>{(rooms?.rooms ?? [room]).map((r) => <option key={r} value={r}>{cap(r)}</option>)}</select></label>
        <div className="swatches">
          {WALLS.map(([n, h]) => <button key={h} type="button" aria-label={n} className={"swatch" + (wall === h ? " on" : "")} style={{ background: h }} onClick={() => setWall(h)} />)}
        </div>
        <label>Budget: <b>{rupees(budget)}</b><input type="range" min={5000} max={300000} step={1000} value={budget} onChange={(e) => setBudget(+e.target.value)} /></label>
        <button className="primary" disabled={busy}>Create board</button>
      </form>
      <form className="card stack" onSubmit={join}>
        <h2>Join with a code</h2>
        <label>Invite code<input value={code} onChange={(e) => setCode(e.target.value)} required /></label>
        <button className="secondary" disabled={busy}>Join</button>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
    </div>
  );
}

/* ---------- assistant ---------- */
type Msg = { role: "user" | "assistant"; text: string };
function AssistantTab({ boards }: { boards: BoardSummary[] }) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [text, setText] = useState(""); const [boardId, setBoardId] = useState<number | "">("");
  const { busy, error, run } = useAsync();
  const send = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;
    const next = [...msgs, { role: "user" as const, text: text.trim() }];
    setMsgs(next); setText("");
    const r = await run(() => api<{ reply: string }>("/assistant/chat", { body: { messages: next, board_id: boardId === "" ? null : boardId } }));
    if (r) setMsgs([...next, { role: "assistant", text: r.reply }]);
  };
  const ideas = ["I have a dusty blue bedroom and ₹40,000. What should I buy first?", "Two sofa ideas for a terracotta wall?"];
  return (
    <div className="stack chat">
      {boards.length > 0 && (
        <label>Talk about a board (optional)
          <select value={boardId} onChange={(e) => setBoardId(e.target.value ? +e.target.value : "")}>
            <option value="">None</option>{boards.map((b) => <option key={b.id} value={b.id}>{b.name}</option>)}
          </select>
        </label>
      )}
      {msgs.length === 0 && (
        <div className="stack">
          <p className="muted">Ask about colours, budgets or what your group has voted for. Answers use the same catalogue and planner as the Plan tab.</p>
          {ideas.map((i) => <button key={i} className="chip" onClick={() => setText(i)}>{i}</button>)}
        </div>
      )}
      {msgs.map((m, i) => <div key={i} className={"bubble " + m.role}>{m.text}</div>)}
      {busy && <div className="bubble assistant muted">Thinking… (can take a little while)</div>}
      {error && <p className="error" role="alert">{error}</p>}
      <form className="row composer" onSubmit={send}>
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Ask anything about your room" maxLength={2000} />
        <button className="primary" disabled={busy}>Send</button>
      </form>
    </div>
  );
}

/* ---------- shell ---------- */
export default function App() {
  const [signedIn, setSignedIn] = useState(!!auth.get());
  const [tab, setTab] = useState<"plan" | "boards" | "assistant">("plan");
  const [rooms, setRooms] = useState<Rooms | null>(null);
  const [boards, setBoards] = useState<BoardSummary[]>([]);
  const [toast, setToast] = useState("");
  const notify = (m: string) => { setToast(m); setTimeout(() => setToast(""), 2200); };
  const reloadBoards = useCallback(() => {
    api<BoardSummary[]>("/boards").then(setBoards).catch((e) => { if (e instanceof ApiError && e.status === 401) { auth.clear(); setSignedIn(false); } });
  }, []);
  useEffect(() => {
    if (!signedIn) return;
    api<Rooms>("/recommend/rooms").then(setRooms).catch(() => {});
    reloadBoards();
  }, [signedIn, reloadBoards]);

  if (!signedIn) return <AuthScreen onDone={() => setSignedIn(true)} />;
  return (
    <div className="app">
      <header><div className="brand">Furnisher<span>Advisor</span></div>
        <button className="link small" onClick={() => { auth.clear(); setSignedIn(false); }}>Sign out</button></header>
      <main>
        {tab === "plan" && <PlanTab rooms={rooms} boards={boards} notify={notify} reloadBoards={reloadBoards} />}
        {tab === "boards" && <BoardsTab rooms={rooms} boards={boards} reload={reloadBoards} notify={notify} />}
        {tab === "assistant" && <AssistantTab boards={boards} />}
      </main>
      {toast && <div className="toast" role="status">{toast}</div>}
      <nav>
        {(["plan", "boards", "assistant"] as const).map((t) => (
          <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{cap(t)}</button>
        ))}
      </nav>
    </div>
  );
}

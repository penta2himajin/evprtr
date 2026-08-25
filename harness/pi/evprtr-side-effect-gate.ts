/**
 * evprtr side-effect gate for Pi (path B)
 *
 * Lets the model call tools freely through the normal Pi agent loop, but
 * requires confirmation before executing side-effect tools.
 *
 * Passthrough (no prompt): read, grep, find, ls
 * Gated: bash, powershell, write, edit
 *
 * Interactive (TUI) or RPC (`ctx.hasUI`): confirm dialog; on Yes the built-in
 * tool runs. Drive RPC confirms with `harness/pi/rpc_bridge.py`.
 *
 * Print/JSON (`-p`, no UI): side-effect tools are blocked.
 *
 * Duplicate suppression: once a write/edit/shell mutation is approved in this
 * Pi session, an identical call is blocked immediately (no second confirm) so
 * rewrite loops cannot burn supervisor time.
 *
 * Optional audit: when EVPRTR_BASE_URL is set (default http://127.0.0.1:8741),
 * each gated call is enqueued at POST /v1/approvals and then approve/reject
 * is recorded to match the UI decision.
 *
 * For path B, disable the compositor response buffer so tool_calls reach Pi:
 *   set EVPRTR_BUFFER_SIDE_EFFECTS=0
 *
 * Usage:
 *   pi -e path/to/evprtr-side-effect-gate.ts --mode rpc --provider evprtr --model evprtr
 *   python harness/pi/rpc_bridge.py run --cwd <repo> --prompt "..."
 */

import { createHash } from "node:crypto";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const GATED = new Set(["bash", "powershell", "write", "edit"]);

const PASSTHROUGH = new Set(["read", "grep", "find", "ls"]);

function baseUrl(): string {
	return (process.env.EVPRTR_BASE_URL || "http://127.0.0.1:8741").replace(/\/$/, "");
}

function sha32(text: string): string {
	return createHash("sha256").update(text, "utf8").digest("hex").slice(0, 32);
}

function mutationFingerprint(toolName: string, input: Record<string, unknown>): string {
	if (toolName === "write") {
		const path = String(input.path ?? "");
		const content = String(input.content ?? "");
		return `write:${path}:${sha32(content)}`;
	}
	if (toolName === "edit") {
		const path = String(input.path ?? "");
		const edits = input.edits ?? [];
		return `edit:${path}:${sha32(JSON.stringify(edits))}`;
	}
	if (toolName === "bash" || toolName === "powershell") {
		return `${toolName}:${sha32(String(input.command ?? ""))}`;
	}
	return `${toolName}:${sha32(JSON.stringify(input))}`;
}

function previewInput(toolName: string, input: Record<string, unknown>): string {
	try {
		if (toolName === "bash" || toolName === "powershell") {
			const cmd = String(input.command ?? "");
			return cmd.length > 500 ? `${cmd.slice(0, 497)}...` : cmd;
		}
		if (toolName === "write") {
			const path = String(input.path ?? "");
			const content = String(input.content ?? "");
			const head = content.length > 280 ? `${content.slice(0, 277)}...` : content;
			return `path=${path} content_len=${content.length}\n---\n${head}`;
		}
		if (toolName === "edit") {
			const path = String(input.path ?? "");
			const edits = Array.isArray(input.edits) ? input.edits : [];
			const parts = edits.slice(0, 3).map((raw, i) => {
				const e = (raw ?? {}) as Record<string, unknown>;
				const oldText = String(e.oldText ?? e.old_string ?? "");
				const newText = String(e.newText ?? e.new_string ?? "");
				const clip = (s: string) => (s.length > 120 ? `${s.slice(0, 117)}...` : s);
				return `#${i} old=${JSON.stringify(clip(oldText))} new=${JSON.stringify(clip(newText))}`;
			});
			const more = edits.length > 3 ? `\n(+${edits.length - 3} more)` : "";
			return `path=${path} edits=${edits.length}\n${parts.join("\n")}${more}`;
		}
		return JSON.stringify(input).slice(0, 500);
	} catch {
		return "(unprintable input)";
	}
}

async function enqueueAudit(payload: {
	tool_name: string;
	arguments: string;
	raw_tool_call: Record<string, unknown>;
	reason: string;
	tags: string[];
}): Promise<string | null> {
	try {
		const res = await fetch(`${baseUrl()}/v1/approvals`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(payload),
		});
		if (!res.ok) return null;
		const data = (await res.json()) as { id?: string };
		return data.id ?? null;
	} catch {
		return null;
	}
}

async function decideAudit(id: string | null, approve: boolean, note: string): Promise<void> {
	if (!id) return;
	const path = approve ? "approve" : "reject";
	try {
		await fetch(`${baseUrl()}/v1/approvals/${id}/${path}`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ note }),
		});
	} catch {
		// audit is best-effort
	}
}

export default function (pi: ExtensionAPI) {
	/** Mutations already approved in this Pi process — block identical retries. */
	const approvedFingerprints = new Set<string>();

	pi.on("tool_call", async (event, ctx) => {
		const name = event.toolName;
		if (PASSTHROUGH.has(name) || !GATED.has(name)) {
			return undefined;
		}

		const input = (event.input ?? {}) as Record<string, unknown>;
		const fp = mutationFingerprint(name, input);
		const summary = previewInput(name, input);
		const argsJson = JSON.stringify(input);

		if (approvedFingerprints.has(fp)) {
			const reason =
				`Duplicate ${name} blocked by evprtr Pi gate: identical mutation ` +
				`already approved this session (fp=${fp}). Do not rewrite the same ` +
				`file with the same content — stop or make a different change.`;
			await enqueueAudit({
				tool_name: name,
				arguments: argsJson,
				raw_tool_call: {
					id: event.toolCallId,
					type: "function",
					function: { name, arguments: argsJson },
				},
				reason: "pi side-effect gate duplicate block",
				tags: ["pi_gate", "side_effect", "duplicate", ctx.mode],
			}).then((id) => decideAudit(id, false, reason));
			return { block: true, reason };
		}

		const actionId = await enqueueAudit({
			tool_name: name,
			arguments: argsJson,
			raw_tool_call: {
				id: event.toolCallId,
				type: "function",
				function: { name, arguments: argsJson },
			},
			reason: "pi side-effect gate (path B)",
			tags: ["pi_gate", "side_effect", ctx.mode],
		});

		if (!ctx.hasUI) {
			await decideAudit(
				actionId,
				false,
				"blocked in non-interactive mode (no UI for confirmation)",
			);
			return {
				block: true,
				reason:
					`Side-effect tool "${name}" blocked by evprtr Pi gate (no UI). ` +
					`Use TUI, or drive RPC confirms via harness/pi/rpc_bridge.py. ` +
					(actionId ? `approval=${actionId}` : "audit unreachable"),
			};
		}

		const title = `evprtr: approve ${name}?`;
		const body =
			`fp=${fp}\n` +
			(actionId ? `[${actionId}]\n` : "") +
			`${summary}\n\nAllow this side-effect tool to run in the Pi harness?`;
		const ok = await ctx.ui.confirm(title, body);
		if (ok) {
			approvedFingerprints.add(fp);
			await decideAudit(actionId, true, "approved via Pi UI/RPC; executed by harness");
			return undefined;
		}
		await decideAudit(actionId, false, "rejected via Pi UI/RPC");
		return { block: true, reason: `Blocked by supervisor (evprtr Pi gate)${actionId ? ` ${actionId}` : ""}` };
	});
}

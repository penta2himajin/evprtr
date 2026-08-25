/**
 * evprtr side-effect gate for Pi (path B)
 *
 * Lets the model call tools freely through the normal Pi agent loop, but
 * requires human confirmation before executing side-effect tools.
 *
 * Passthrough (no prompt): read, grep, find, ls
 * Gated: bash, powershell, write, edit
 *
 * Interactive (TUI/RPC): confirm dialog; on Yes the built-in tool runs.
 * Print/JSON (-p): side-effect tools are blocked (no UI). Prefer read-only
 * tools for unattended plan turns.
 *
 * Optional audit: when EVPRTR_BASE_URL is set (default http://127.0.0.1:8741),
 * each gated call is enqueued at POST /v1/approvals and then approve/reject
 * is recorded to match the UI decision.
 *
 * For path B, disable the compositor response buffer so tool_calls reach Pi:
 *   set EVPRTR_BUFFER_SIDE_EFFECTS=0
 *
 * Usage:
 *   pi -e path/to/evprtr-side-effect-gate.ts --provider evprtr --model evprtr
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const GATED = new Set(["bash", "powershell", "write", "edit"]);

const PASSTHROUGH = new Set(["read", "grep", "find", "ls"]);

function baseUrl(): string {
	return (process.env.EVPRTR_BASE_URL || "http://127.0.0.1:8741").replace(/\/$/, "");
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
			return `path=${path} content_len=${content.length}`;
		}
		if (toolName === "edit") {
			const path = String(input.path ?? "");
			const edits = Array.isArray(input.edits) ? input.edits.length : 0;
			return `path=${path} edits=${edits}`;
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
	pi.on("tool_call", async (event, ctx) => {
		const name = event.toolName;
		if (PASSTHROUGH.has(name) || !GATED.has(name)) {
			return undefined;
		}

		const input = (event.input ?? {}) as Record<string, unknown>;
		const summary = previewInput(name, input);
		const argsJson = JSON.stringify(input);
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
					`Use read/grep/find/ls, or run interactively to approve. ` +
					(actionId ? `approval=${actionId}` : "audit unreachable"),
			};
		}

		const title = `evprtr: approve ${name}?`;
		const body =
			(actionId ? `[${actionId}]\n` : "") +
			`${summary}\n\nAllow this side-effect tool to run in the Pi harness?`;
		const ok = await ctx.ui.confirm(title, body);
		if (ok) {
			await decideAudit(actionId, true, "approved in Pi UI; executed by harness");
			return undefined;
		}
		await decideAudit(actionId, false, "rejected in Pi UI");
		return { block: true, reason: `Blocked by user (evprtr Pi gate)${actionId ? ` ${actionId}` : ""}` };
	});
}

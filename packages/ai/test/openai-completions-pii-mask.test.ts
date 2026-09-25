import { describe, expect, it } from "vitest";
import { getModel } from "../src/models.js";
import { convertMessages } from "../src/providers/openai-completions.js";
import type {
	AssistantMessage,
	Context,
	Model,
	OpenAICompletionsCompat,
	ToolResultMessage,
	Usage,
} from "../src/types.js";

const emptyUsage: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const compat = {
	supportsStore: true,
	supportsDeveloperRole: true,
	supportsReasoningEffort: true,
	supportsUsageInStreaming: true,
	maxTokensField: "max_completion_tokens",
	requiresToolResultName: false,
	requiresAssistantAfterToolResult: false,
	requiresThinkingAsText: false,
	requiresReasoningContentOnAssistantMessages: false,
	thinkingFormat: "openai",
	openRouterRouting: {},
	vercelGatewayRouting: {},
	zaiToolStream: false,
	supportsStrictMode: true,
	cacheControlFormat: undefined,
	sendSessionAffinityHeaders: false,
	supportsLongCacheRetention: true,
	preserveThinking: false,
	enableSearch: false,
	searchStrategy: undefined,
	forcedSearch: false,
} satisfies Required<
	Omit<
		OpenAICompletionsCompat,
		"cacheControlFormat" | "searchStrategy" | "toolStream" | "reasoningCountsTowardMaxTokens"
	>
> & {
	cacheControlFormat?: OpenAICompletionsCompat["cacheControlFormat"];
	searchStrategy?: OpenAICompletionsCompat["searchStrategy"];
	toolStream?: boolean;
	reasoningCountsTowardMaxTokens?: boolean;
};

const makeModel = (): Model<"openai-completions"> => {
	const { compat: _compat, ...baseModel } = getModel("openai", "gpt-4o-mini")!;
	return { ...baseModel, api: "openai-completions" };
};

const convert = (context: Context): Array<{ role: string; content: unknown }> =>
	convertMessages(makeModel(), context, compat) as Array<{ role: string; content: unknown }>;

const assistantWithToolCall = (now: number): AssistantMessage => ({
	role: "assistant",
	content: [{ type: "toolCall", id: "call-1", name: "read", arguments: { path: "roster.txt" } }],
	api: "openai-completions",
	provider: "openai",
	model: "gpt-4o-mini",
	usage: emptyUsage,
	stopReason: "toolUse",
	timestamp: now,
});

describe("openai-completions PII hard gate", () => {
	it("masks mainland mobile numbers in user string messages", () => {
		const params = convert({
			messages: [{ role: "user", content: "水务台账:张三 13812345678,李四 15987654321", timestamp: Date.now() }],
		});
		const user = params.find((m) => m.role === "user")!;
		expect(user.content).toBe("水务台账:张三 138****5678,李四 159****4321");
	});

	it("masks mobile numbers in the system prompt", () => {
		const params = convert({
			systemPrompt: "客户经理手机号 13812345678,有事直接联系",
			messages: [{ role: "user", content: "hi", timestamp: Date.now() }],
		});
		const system = params.find((m) => m.role === "system")!;
		expect(system.content).toContain("138****5678");
		expect(system.content).not.toContain("13812345678");
	});

	it("masks mobile numbers in tool result text (the roster ingestion path)", () => {
		const now = Date.now();
		const toolResult: ToolResultMessage = {
			role: "toolResult",
			toolCallId: "call-1",
			toolName: "read",
			content: [{ type: "text", text: "row1: 王五 13612345678\nrow2: 赵六 18876543210" }],
			isError: false,
			timestamp: now,
		};
		const params = convert({
			messages: [{ role: "user", content: "读台账", timestamp: now - 1 }, assistantWithToolCall(now), toolResult],
		});
		const tool = params.find((m) => m.role === "tool")!;
		const content = String(tool.content);
		expect(content).toContain("136****5678");
		expect(content).toContain("188****3210");
		expect(content).not.toContain("13612345678");
		expect(content).not.toContain("18876543210");
	});

	it("masks 18-digit ID card numbers", () => {
		const params = convert({
			messages: [{ role: "user", content: "身份证 11010119900307891X", timestamp: Date.now() }],
		});
		const user = params.find((m) => m.role === "user")!;
		expect(user.content).toBe("身份证 110101********891X");
	});

	it("leaves long digit runs (order ids) untouched", () => {
		const params = convert({
			messages: [{ role: "user", content: "订单号 2026092412345678901 请查询", timestamp: Date.now() }],
		});
		const user = params.find((m) => m.role === "user")!;
		expect(user.content).toBe("订单号 2026092412345678901 请查询");
	});

	it("PRIME_PII_MASK=off restores raw passthrough", () => {
		process.env.PRIME_PII_MASK = "off";
		try {
			const params = convert({
				messages: [{ role: "user", content: "手机号 13812345678", timestamp: Date.now() }],
			});
			const user = params.find((m) => m.role === "user")!;
			expect(user.content).toBe("手机号 13812345678");
		} finally {
			delete process.env.PRIME_PII_MASK;
		}
	});
});

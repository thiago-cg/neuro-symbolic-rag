import { z } from "zod";

export const ResearchBriefSchema = z.object({
  domain: z.string(),
  intent: z.string(),
  subTopics: z.array(z.string()),
});

export type ResearchBrief = z.infer<typeof ResearchBriefSchema>;

export const SubAgentFindingsSchema = z.object({
  subTopic: z.string(),
  paperIds: z.array(z.string()),
});

export type SubAgentFindings = z.infer<typeof SubAgentFindingsSchema>;

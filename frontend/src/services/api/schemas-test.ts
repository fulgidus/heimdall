// Test file to verify schemas compilation
import { z } from 'zod';

export const TestSchema = z.object({
  test_nullable: z.string().nullable().optional(),
  test_normal: z.string().optional(),
});

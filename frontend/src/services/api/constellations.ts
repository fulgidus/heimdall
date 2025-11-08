/**
 * Constellations API Service
 *
 * Handles all constellation-related API calls:
 * - CRUD operations for constellations
 * - Managing constellation members (WebSDR assignments)
 * - Sharing constellations with other users
 */

import { z } from 'zod';
import api from '@/lib/api';

// ============================================================================
// Zod Schemas
// ============================================================================

export const ConstellationMemberSchema = z.object({
    websdr_station_id: z.string().uuid(),
    added_at: z.string(),
    added_by: z.string().nullish(),
});

export const ConstellationShareSchema = z.object({
    id: z.string().uuid(),
    constellation_id: z.string().uuid(),
    user_id: z.string(),
    permission: z.enum(['read', 'edit']),
    shared_by: z.string(),
    shared_at: z.string(),
});

export const ConstellationSchema = z.object({
    id: z.string().uuid(),
    name: z.string(),
    description: z.string().nullish(),
    owner_id: z.string(),
    created_at: z.string(),
    updated_at: z.string(),
    member_count: z.number().nonnegative(),
    members: z.array(ConstellationMemberSchema).optional(),
    shares: z.array(ConstellationShareSchema).optional(),
});

export const CreateConstellationSchema = z.object({
    name: z.string().min(1).max(255),
    description: z.string().optional(),
});

export const UpdateConstellationSchema = z.object({
    name: z.string().min(1).max(255).optional(),
    description: z.string().optional(),
});

export const AddMemberSchema = z.object({
    websdr_station_id: z.string().uuid(),
});

export const CreateShareSchema = z.object({
    user_id: z.string().min(1),
    permission: z.enum(['read', 'edit']),
});

export const UpdateShareSchema = z.object({
    permission: z.enum(['read', 'edit']),
});

// ============================================================================
// TypeScript Types
// ============================================================================

export type ConstellationMember = z.infer<typeof ConstellationMemberSchema>;
export type ConstellationShare = z.infer<typeof ConstellationShareSchema>;
export type Constellation = z.infer<typeof ConstellationSchema>;
export type CreateConstellationRequest = z.infer<typeof CreateConstellationSchema>;
export type UpdateConstellationRequest = z.infer<typeof UpdateConstellationSchema>;
export type AddMemberRequest = z.infer<typeof AddMemberSchema>;
export type CreateShareRequest = z.infer<typeof CreateShareSchema>;
export type UpdateShareRequest = z.infer<typeof UpdateShareSchema>;

// ============================================================================
// API Service Functions
// ============================================================================

/**
 * Get list of all constellations accessible to the current user
 * (owned + shared)
 */
export async function getConstellations(): Promise<Constellation[]> {
    console.log('🌌 ConstellationsService.getConstellations(): calling GET /v1/constellations');
    const response = await api.get('/v1/constellations');

    // Handle potential data wrapper
    let data = response.data;
    if (data && typeof data === 'object' && 'data' in data) {
        data = data.data;
    }

    if (!Array.isArray(data)) {
        console.error('❌ getConstellations(): Response is not an array. Received:', data);
        throw new Error(`Expected array of constellations, got ${typeof data}`);
    }

    // Validate response with Zod
    try {
        const validated = z.array(ConstellationSchema).parse(data);
        console.log('✅ ConstellationsService.getConstellations(): received', validated.length, 'constellations');
        return validated;
    } catch (zodError) {
        console.error('❌ Zod validation error in getConstellations():', zodError);
        if (zodError instanceof z.ZodError) {
            throw new Error(`Constellation validation error: ${zodError.message}`);
        }
        throw zodError;
    }
}

/**
 * Get details for a specific constellation (including members and shares)
 */
export async function getConstellation(id: string): Promise<Constellation> {
    console.log(`🌌 ConstellationsService.getConstellation(${id})`);
    const response = await api.get(`/v1/constellations/${id}`);

    // Handle potential data wrapper
    let data = response.data;
    if (data && typeof data === 'object' && 'data' in data) {
        data = data.data;
    }

    // Validate response with Zod
    try {
        const validated = ConstellationSchema.parse(data);
        console.log('✅ ConstellationsService.getConstellation(): retrieved constellation', validated.name);
        return validated;
    } catch (zodError) {
        console.error('❌ Zod validation error in getConstellation():', zodError);
        if (zodError instanceof z.ZodError) {
            throw new Error(`Constellation validation error: ${zodError.message}`);
        }
        throw zodError;
    }
}

/**
 * Create a new constellation (operator+ only)
 */
export async function createConstellation(data: CreateConstellationRequest): Promise<Constellation> {
    console.log('🌌 ConstellationsService.createConstellation():', data);
    
    // Validate request data
    CreateConstellationSchema.parse(data);
    
    const response = await api.post('/v1/constellations', data);

    // Handle potential data wrapper
    let responseData = response.data;
    if (responseData && typeof responseData === 'object' && 'data' in responseData) {
        responseData = responseData.data;
    }

    // Validate response with Zod
    try {
        const validated = ConstellationSchema.parse(responseData);
        console.log('✅ ConstellationsService.createConstellation(): created constellation', validated.id);
        return validated;
    } catch (zodError) {
        console.error('❌ Zod validation error in createConstellation():', zodError);
        if (zodError instanceof z.ZodError) {
            throw new Error(`Constellation validation error: ${zodError.message}`);
        }
        throw zodError;
    }
}

/**
 * Update an existing constellation (owner or edit permission required)
 */
export async function updateConstellation(id: string, data: UpdateConstellationRequest): Promise<Constellation> {
    console.log(`🌌 ConstellationsService.updateConstellation(${id}):`, data);
    
    // Validate request data
    UpdateConstellationSchema.parse(data);
    
    const response = await api.put(`/v1/constellations/${id}`, data);

    // Handle potential data wrapper
    let responseData = response.data;
    if (responseData && typeof responseData === 'object' && 'data' in responseData) {
        responseData = responseData.data;
    }

    // Validate response with Zod
    try {
        const validated = ConstellationSchema.parse(responseData);
        console.log('✅ ConstellationsService.updateConstellation(): updated constellation', validated.id);
        return validated;
    } catch (zodError) {
        console.error('❌ Zod validation error in updateConstellation():', zodError);
        if (zodError instanceof z.ZodError) {
            throw new Error(`Constellation validation error: ${zodError.message}`);
        }
        throw zodError;
    }
}

/**
 * Delete a constellation (owner only)
 */
export async function deleteConstellation(id: string): Promise<void> {
    console.log(`🌌 ConstellationsService.deleteConstellation(${id})`);
    await api.delete(`/v1/constellations/${id}`);
    console.log('✅ ConstellationsService.deleteConstellation(): deleted constellation', id);
}

/**
 * Add a WebSDR to a constellation (owner or edit permission required)
 */
export async function addConstellationMember(constellationId: string, data: AddMemberRequest): Promise<ConstellationMember> {
    console.log(`🌌 ConstellationsService.addConstellationMember(${constellationId}):`, data);
    
    // Validate request data
    AddMemberSchema.parse(data);
    
    const response = await api.post(`/v1/constellations/${constellationId}/members`, data);

    // Handle potential data wrapper
    let responseData = response.data;
    if (responseData && typeof responseData === 'object' && 'data' in responseData) {
        responseData = responseData.data;
    }

    // Validate response with Zod
    try {
        const validated = ConstellationMemberSchema.parse(responseData);
        console.log('✅ ConstellationsService.addConstellationMember(): added member', validated.websdr_station_id);
        return validated;
    } catch (zodError) {
        console.error('❌ Zod validation error in addConstellationMember():', zodError);
        if (zodError instanceof z.ZodError) {
            throw new Error(`Constellation member validation error: ${zodError.message}`);
        }
        throw zodError;
    }
}

/**
 * Remove a WebSDR from a constellation (owner or edit permission required)
 */
export async function removeConstellationMember(constellationId: string, websdrId: string): Promise<void> {
    console.log(`🌌 ConstellationsService.removeConstellationMember(${constellationId}, ${websdrId})`);
    await api.delete(`/v1/constellations/${constellationId}/members/${websdrId}`);
    console.log('✅ ConstellationsService.removeConstellationMember(): removed member', websdrId);
}

/**
 * Get list of shares for a constellation
 */
export async function getConstellationShares(constellationId: string): Promise<ConstellationShare[]> {
    console.log(`🌌 ConstellationsService.getConstellationShares(${constellationId})`);
    const response = await api.get(`/v1/constellations/${constellationId}/shares`);

    // Handle potential data wrapper
    let data = response.data;
    if (data && typeof data === 'object' && 'data' in data) {
        data = data.data;
    }

    if (!Array.isArray(data)) {
        console.error('❌ getConstellationShares(): Response is not an array. Received:', data);
        throw new Error(`Expected array of shares, got ${typeof data}`);
    }

    // Validate response with Zod
    try {
        const validated = z.array(ConstellationShareSchema).parse(data);
        console.log('✅ ConstellationsService.getConstellationShares(): received', validated.length, 'shares');
        return validated;
    } catch (zodError) {
        console.error('❌ Zod validation error in getConstellationShares():', zodError);
        if (zodError instanceof z.ZodError) {
            throw new Error(`Share validation error: ${zodError.message}`);
        }
        throw zodError;
    }
}

/**
 * Share a constellation with another user (owner only)
 */
export async function createConstellationShare(constellationId: string, data: CreateShareRequest): Promise<ConstellationShare> {
    console.log(`🌌 ConstellationsService.createConstellationShare(${constellationId}):`, data);
    
    // Validate request data
    CreateShareSchema.parse(data);
    
    const response = await api.post(`/v1/constellations/${constellationId}/shares`, data);

    // Handle potential data wrapper
    let responseData = response.data;
    if (responseData && typeof responseData === 'object' && 'data' in responseData) {
        responseData = responseData.data;
    }

    // Validate response with Zod
    try {
        const validated = ConstellationShareSchema.parse(responseData);
        console.log('✅ ConstellationsService.createConstellationShare(): created share', validated.id);
        return validated;
    } catch (zodError) {
        console.error('❌ Zod validation error in createConstellationShare():', zodError);
        if (zodError instanceof z.ZodError) {
            throw new Error(`Share validation error: ${zodError.message}`);
        }
        throw zodError;
    }
}

/**
 * Update permission level for an existing share (owner only)
 */
export async function updateConstellationShare(constellationId: string, userId: string, data: UpdateShareRequest): Promise<ConstellationShare> {
    console.log(`🌌 ConstellationsService.updateConstellationShare(${constellationId}, ${userId}):`, data);
    
    // Validate request data
    UpdateShareSchema.parse(data);
    
    const response = await api.put(`/v1/constellations/${constellationId}/shares/${userId}`, data);

    // Handle potential data wrapper
    let responseData = response.data;
    if (responseData && typeof responseData === 'object' && 'data' in responseData) {
        responseData = responseData.data;
    }

    // Validate response with Zod
    try {
        const validated = ConstellationShareSchema.parse(responseData);
        console.log('✅ ConstellationsService.updateConstellationShare(): updated share', validated.id);
        return validated;
    } catch (zodError) {
        console.error('❌ Zod validation error in updateConstellationShare():', zodError);
        if (zodError instanceof z.ZodError) {
            throw new Error(`Share validation error: ${zodError.message}`);
        }
        throw zodError;
    }
}

/**
 * Remove a share (unshare with user) (owner only)
 */
export async function deleteConstellationShare(constellationId: string, userId: string): Promise<void> {
    console.log(`🌌 ConstellationsService.deleteConstellationShare(${constellationId}, ${userId})`);
    await api.delete(`/v1/constellations/${constellationId}/shares/${userId}`);
    console.log('✅ ConstellationsService.deleteConstellationShare(): removed share for user', userId);
}

// Default export for convenience
export default {
    getConstellations,
    getConstellation,
    createConstellation,
    updateConstellation,
    deleteConstellation,
    addConstellationMember,
    removeConstellationMember,
    getConstellationShares,
    createConstellationShare,
    updateConstellationShare,
    deleteConstellationShare,
};

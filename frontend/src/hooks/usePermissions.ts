import { useAuth } from './useAuth';

/**
 * Permission types for shared resources
 */
export type Permission = 'read' | 'edit';

/**
 * Resource types that can be owned and shared
 */
export type Resource = 'constellation' | 'source' | 'model';

/**
 * Resource object with ownership and sharing information
 */
interface OwnedResource {
  id: string;
  owner_id: string;
  shares?: Array<{
    user_id: string;
    permission: Permission;
  }>;
}

/**
 * Hook for resource-level permission checks.
 * Implements RBAC permission matrix for Heimdall.
 * 
 * Permission Matrix:
 * - ADMIN: Full access to all resources (bypasses all checks)
 * - OPERATOR: Can CRUD owned/shared resources (if shared with 'edit')
 * - USER: Can view assigned resources (if shared with 'read' or 'edit')
 * 
 * Usage:
 * const { canViewConstellation, canEditConstellation, canDeleteConstellation } = usePermissions();
 * 
 * if (canEditConstellation(constellation)) {
 *   // Show edit button
 * }
 */
export const usePermissions = () => {
  const { user, isAdmin, isOperator } = useAuth();

  /**
   * Check if user can view a resource (read access)
   * @param resource - The resource to check (constellation, source, or model)
   * @returns true if user has read access
   */
  const canView = (resource: OwnedResource): boolean => {
    // Admin can view everything
    if (isAdmin) return true;

    // Owner can view their own resources
    if (user?.id === resource.owner_id) return true;

    // Check if resource is shared with user (read or edit permission)
    const share = resource.shares?.find(s => s.user_id === user?.id);
    return !!share; // Any permission level grants read access
  };

  /**
   * Check if user can edit a resource
   * @param resource - The resource to check
   * @returns true if user has edit access
   */
  const canEdit = (resource: OwnedResource): boolean => {
    // Admin can edit everything
    if (isAdmin) return true;

    // Owner can edit their own resources
    if (user?.id === resource.owner_id) return true;

    // Non-operators cannot edit even if shared
    if (!isOperator) return false;

    // Operator can edit if shared with 'edit' permission
    const share = resource.shares?.find(s => s.user_id === user?.id);
    return share?.permission === 'edit';
  };

  /**
   * Check if user can delete a resource
   * @param resource - The resource to check
   * @returns true if user has delete access
   */
  const canDelete = (resource: OwnedResource): boolean => {
    // Admin can delete everything
    if (isAdmin) return true;

    // Only owner can delete (sharing doesn't grant delete permission)
    return user?.id === resource.owner_id;
  };

  /**
   * Check if user can share a resource with others
   * @param resource - The resource to check
   * @returns true if user can share the resource
   */
  const canShare = (resource: OwnedResource): boolean => {
    // Admin can share everything
    if (isAdmin) return true;

    // Only owner can share their resources
    return user?.id === resource.owner_id;
  };

  /**
   * Check if user can create a resource type
   * @param resourceType - Type of resource (constellation, source, model)
   * @returns true if user can create this resource type
   */
  const canCreate = (resourceType: Resource): boolean => {
    // Admin can create everything
    if (isAdmin) return true;

    // Operators can create constellations, sources, and models
    if (isOperator) return true;

    // Users cannot create resources
    return false;
  };

  // Convenience methods for specific resource types

  /**
   * Constellation permissions
   */
  const canViewConstellation = (constellation: OwnedResource) => canView(constellation);
  const canEditConstellation = (constellation: OwnedResource) => canEdit(constellation);
  const canDeleteConstellation = (constellation: OwnedResource) => canDelete(constellation);
  const canShareConstellation = (constellation: OwnedResource) => canShare(constellation);
  const canCreateConstellation = () => canCreate('constellation');

  /**
   * Source permissions
   */
  const canViewSource = (source: OwnedResource) => canView(source);
  const canEditSource = (source: OwnedResource) => canEdit(source);
  const canDeleteSource = (source: OwnedResource) => canDelete(source);
  const canShareSource = (source: OwnedResource) => canShare(source);
  const canCreateSource = () => canCreate('source');

  /**
   * Model permissions
   */
  const canViewModel = (model: OwnedResource) => canView(model);
  const canEditModel = (model: OwnedResource) => canEdit(model);
  const canDeleteModel = (model: OwnedResource) => canDelete(model);
  const canShareModel = (model: OwnedResource) => canShare(model);
  const canCreateModel = () => canCreate('model');

  /**
   * Special permissions
   */
  const canAccessSystemSettings = () => isAdmin;
  const canManageUsers = () => isAdmin;
  const canStartTraining = () => isOperator || isAdmin;
  const canStartAcquisition = () => isOperator || isAdmin;
  const canGenerateSyntheticData = () => isOperator || isAdmin;

  return {
    // Generic methods
    canView,
    canEdit,
    canDelete,
    canShare,
    canCreate,

    // Constellation-specific
    canViewConstellation,
    canEditConstellation,
    canDeleteConstellation,
    canShareConstellation,
    canCreateConstellation,

    // Source-specific
    canViewSource,
    canEditSource,
    canDeleteSource,
    canShareSource,
    canCreateSource,

    // Model-specific
    canViewModel,
    canEditModel,
    canDeleteModel,
    canShareModel,
    canCreateModel,

    // Special permissions
    canAccessSystemSettings,
    canManageUsers,
    canStartTraining,
    canStartAcquisition,
    canGenerateSyntheticData,
  };
};

export default usePermissions;

# Heimdall RBAC User Guide

## Overview

Heimdall implements **Role-Based Access Control (RBAC)** to manage user permissions and resource access. This guide explains how to use the RBAC system as a user, operator, or administrator.

## Table of Contents

- [Roles and Permissions](#roles-and-permissions)
- [Constellations](#constellations)
- [Sharing Resources](#sharing-resources)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

---

## Roles and Permissions

Heimdall uses three roles with hierarchical permissions:

### 👤 USER

Basic access for viewing and using assigned resources.

**What you can do**:
- View constellations shared with you
- Start localization sessions on assigned constellations
- Configure frequency and session parameters
- View session history for assigned constellations
- View system status

**What you cannot do**:
- Create constellations, sources, or models
- Edit or delete any resources
- Share resources with other users
- Access system settings or admin functions

**Typical Use Case**: Field operators or stakeholders who need to monitor specific constellations and start localization sessions.

### 🛠️ OPERATOR

Advanced access for creating and managing resources.

**What you can do**:
- **Everything a USER can do, plus**:
- Create new constellations (you become the owner)
- Edit constellations you own or have been shared with 'edit' permission
- Delete constellations you own
- Add/remove WebSDR stations to/from your constellations
- Share your constellations with other users
- Create and manage sources (transmitters/beacons)
- Create and manage ML models
- Start RF acquisitions
- Start training sessions
- Generate synthetic samples

**What you cannot do**:
- Access resources owned by others unless shared
- View or edit system settings
- Manage user accounts

**Typical Use Case**: RF engineers, data scientists, or team leads who need to create and manage constellations and resources for their projects.

### 🔑 ADMIN

Full system access with no restrictions.

**What you can do**:
- **Everything an OPERATOR can do, plus**:
- View and edit ALL constellations, sources, and models (bypasses ownership)
- Delete any resource
- Manage user accounts and assignments
- Modify system settings
- Access all admin functions

**Typical Use Case**: System administrators responsible for maintaining the platform and managing users.

---

## Constellations

### What is a Constellation?

A **Constellation** is a logical grouping of WebSDR stations used for radio source localization. Constellations allow you to:

- **Organize** WebSDR stations by geography, frequency band, or project
- **Control Access** to specific stations for different users
- **Collaborate** by sharing constellations with read or edit permissions
- **Isolate** resources for different teams or use cases

**Example Constellations**:
- "Northern Italy Coverage" - Stations covering northern Italy
- "VHF Monitoring Network" - Stations optimized for VHF frequencies
- "Team Alpha Resources" - Stations assigned to Team Alpha

### Creating a Constellation (OPERATOR+)

1. Navigate to **Constellations** page
2. Click **Create Constellation** button
3. Fill in the form:
   - **Name**: Short, descriptive name (e.g., "Northern Italy Coverage")
   - **Description**: Optional detailed description
4. Click **Create**

**Result**: You are now the owner of this constellation and can add WebSDR stations to it.

### Adding WebSDR Stations to a Constellation (OPERATOR+)

1. Open the constellation details page
2. Click **Add WebSDR** button
3. Select one or more WebSDR stations from the list
4. Click **Add Selected**

**Note**: All WebSDR stations are globally visible to authenticated users. You're assigning them to your constellation for localization purposes.

### Editing a Constellation

**Requirements**: Owner OR shared with 'edit' permission OR admin

1. Open the constellation details page
2. Click **Edit** button
3. Modify name or description
4. Click **Save Changes**

### Deleting a Constellation

**Requirements**: Owner OR admin only

1. Open the constellation details page
2. Click **Delete** button (red)
3. Confirm deletion in the dialog

**Warning**: Deleting a constellation also removes:
- All WebSDR station assignments
- All shares with other users
- This action cannot be undone

---

## Sharing Resources

Owners can share their constellations, sources, and models with other users.

### Permission Levels

- **Read**: User can view the resource and use it (e.g., start sessions on a constellation)
- **Edit**: User can view AND modify the resource (e.g., add/remove WebSDRs, change name/description)

**Owner Privileges** (beyond 'edit'):
- Delete the resource
- Share the resource with other users
- Modify or revoke sharing permissions

### Sharing a Constellation

**Requirements**: Owner OR admin only

1. Open the constellation details page
2. Click **Share** button
3. In the Share modal:
   - **Search for users** by email or username (type at least 3 characters)
   - Select a user from the dropdown
   - Choose permission level: **Read** or **Edit**
   - Click **Add Share**
4. The user now has access to your constellation

### Managing Shares

**View existing shares**:
1. Open the constellation details page
2. Click **Share** button
3. The modal shows all current shares with user avatars and permissions

**Update permission**:
1. In the Share modal, find the user
2. Click the permission dropdown next to their name
3. Select new permission level (Read or Edit)

**Remove share**:
1. In the Share modal, find the user
2. Click the **Remove** (trash icon) button next to their name
3. Confirm removal

### Viewing Shared Resources

**As a USER or OPERATOR**:

- Resources shared with you appear in the respective list pages
- Each card shows:
  - **Owner badge** if you own it
  - **Permission badge** (Read/Edit) if shared with you
  - Available actions based on your permission level

**Example**: A constellation shared with 'Read' permission shows a **Read** badge and no edit/delete buttons.

---

## Common Workflows

### Workflow 1: Starting a Localization Session (USER+)

1. Navigate to **Localization** page
2. Select a constellation from the dropdown (shows only assigned constellations)
3. Configure session parameters:
   - Frequency (MHz)
   - Bandwidth
   - Duration
4. Click **Start Session**
5. Monitor real-time localization results on the map

### Workflow 2: Creating and Sharing a Constellation (OPERATOR+)

**Scenario**: You want to create a constellation for your team and share it with team members.

1. **Create Constellation**:
   - Go to **Constellations** page
   - Click **Create Constellation**
   - Name: "Team Alpha VHF Network"
   - Description: "VHF monitoring stations for Team Alpha"
   - Click **Create**

2. **Add WebSDR Stations**:
   - Open the new constellation
   - Click **Add WebSDR**
   - Select stations: "Rome-IT", "Milan-IT", "Bologna-IT"
   - Click **Add Selected**

3. **Share with Team Members**:
   - Click **Share** button
   - Search for team member: "alice@example.com"
   - Permission: **Edit** (allow Alice to manage stations)
   - Click **Add Share**
   - Repeat for other team members with **Read** permission

4. **Result**: Team members can now:
   - View the constellation (all members)
   - Start localization sessions (all members)
   - Add/remove WebSDR stations (Alice only)

### Workflow 3: Managing Sources as an Operator (OPERATOR+)

**Scenario**: You want to track known transmitters in your area.

1. **Create a Source**:
   - Go to **Sources** page
   - Click **Create Source**
   - Fill in details:
     - Name: "Beacon-Milano-VHF"
     - Frequency: 145.850 MHz
     - Location: Latitude 45.4642, Longitude 9.1900
     - Type: Beacon
     - Description: "VHF beacon in Milano"
   - Click **Create**

2. **Share the Source** (optional):
   - Open source details
   - Click **Share**
   - Add team members with Read or Edit permissions

3. **Use in Training**:
   - The source is now available when generating synthetic training data
   - Can be used as ground truth for model evaluation

### Workflow 4: Admin Managing All Resources (ADMIN)

**Scenario**: As admin, you need to audit all constellations and fix a misconfigured one.

1. **View All Constellations**:
   - Go to **Constellations** page
   - You see ALL constellations (not just yours)
   - Each shows the owner's name

2. **Edit Any Constellation**:
   - Open any constellation (regardless of owner)
   - Click **Edit** (available even if you don't own it)
   - Make changes and save

3. **Delete Problematic Constellation**:
   - Open the constellation
   - Click **Delete**
   - Confirm deletion

**Note**: Admins bypass all ownership and sharing checks. Use this power responsibly!

---

## Troubleshooting

### "Access Denied" or 403 Forbidden Error

**Possible Causes**:

1. **Insufficient Role**:
   - You're trying to access a resource that requires a higher role
   - **Solution**: Contact your admin to upgrade your role

2. **Not Owner/Not Shared**:
   - You're trying to edit/delete a resource you don't own or that hasn't been shared with you
   - **Solution**: Contact the owner to request 'edit' permission or ask an admin

3. **Wrong Permission Level**:
   - You have 'read' permission but are trying to edit
   - **Solution**: Ask the owner to upgrade your permission to 'edit'

4. **Session Expired**:
   - Your JWT token has expired
   - **Solution**: Refresh the page to re-authenticate

### Cannot See a Constellation

**Possible Causes**:

1. **Not Shared with You**:
   - The constellation exists but hasn't been shared with you
   - **Solution**: Ask the owner to share it with you

2. **Role Restriction**:
   - As a USER, you can only see constellations shared with you
   - **Solution**: This is expected behavior. Request access from the owner

3. **Deleted Constellation**:
   - The constellation was deleted by the owner or admin
   - **Solution**: No recovery possible. Create a new one

### Cannot Add WebSDR to Constellation

**Possible Causes**:

1. **No Edit Permission**:
   - You need 'edit' permission or ownership to add WebSDRs
   - **Solution**: Ask owner for 'edit' permission

2. **WebSDR Already in Constellation**:
   - The WebSDR is already a member of this constellation
   - **Solution**: This is expected. Each WebSDR can only be added once

3. **WebSDR Offline**:
   - The WebSDR station is currently offline (shouldn't prevent adding, but check status)
   - **Solution**: Verify WebSDR status on **WebSDRs** page

### Share Button Not Visible

**Possible Causes**:

1. **Not Owner**:
   - Only owners can share resources
   - **Solution**: You cannot share resources you don't own

2. **Insufficient Role**:
   - Only OPERATOR+ can create and therefore own resources
   - **Solution**: Request role upgrade from admin

---

## API Reference

### Authentication

All API requests require a valid JWT token from Keycloak:

```bash
Authorization: Bearer <JWT_TOKEN>
```

### Constellation Endpoints

#### List Constellations

```http
GET /api/v1/constellations
```

**Query Parameters**:
- `skip` (optional): Pagination offset (default: 0)
- `limit` (optional): Number of results (default: 100, max: 1000)

**Response**:
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Northern Italy Coverage",
    "description": "Stations covering northern Italy",
    "owner_id": "user-123",
    "created_at": "2025-01-15T10:30:00Z",
    "updated_at": "2025-01-15T10:30:00Z",
    "member_count": 5,
    "permission": "edit"  // or "read" or null (if owner)
  }
]
```

**Filtering**:
- **USER**: Returns only constellations shared with you
- **OPERATOR**: Returns constellations you own or have been shared with you
- **ADMIN**: Returns ALL constellations

#### Create Constellation

```http
POST /api/v1/constellations
```

**Requirements**: OPERATOR or ADMIN role

**Request Body**:
```json
{
  "name": "Northern Italy Coverage",
  "description": "Stations covering northern Italy"
}
```

**Response**: `201 Created`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Northern Italy Coverage",
  "description": "Stations covering northern Italy",
  "owner_id": "user-123",
  "created_at": "2025-01-15T10:30:00Z",
  "updated_at": "2025-01-15T10:30:00Z"
}
```

#### Get Constellation Details

```http
GET /api/v1/constellations/{constellation_id}
```

**Requirements**: Owner, shared user, or admin

**Response**: `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Northern Italy Coverage",
  "description": "Stations covering northern Italy",
  "owner_id": "user-123",
  "created_at": "2025-01-15T10:30:00Z",
  "updated_at": "2025-01-15T10:30:00Z",
  "members": [
    {
      "id": "websdr-001",
      "name": "Rome-IT",
      "location": "Rome, Italy",
      "added_at": "2025-01-15T10:35:00Z"
    }
  ]
}
```

**Error Responses**:
- `403 Forbidden`: No permission to view this constellation
- `404 Not Found`: Constellation does not exist

#### Update Constellation

```http
PUT /api/v1/constellations/{constellation_id}
```

**Requirements**: Owner, shared with 'edit' permission, or admin

**Request Body**:
```json
{
  "name": "Updated Name",
  "description": "Updated description"
}
```

**Response**: `200 OK` (updated constellation object)

**Error Responses**:
- `403 Forbidden`: No edit permission
- `404 Not Found`: Constellation does not exist

#### Delete Constellation

```http
DELETE /api/v1/constellations/{constellation_id}
```

**Requirements**: Owner or admin only

**Response**: `204 No Content`

**Error Responses**:
- `403 Forbidden`: Not owner or admin
- `404 Not Found`: Constellation does not exist

#### Add WebSDR to Constellation

```http
POST /api/v1/constellations/{constellation_id}/members
```

**Requirements**: Owner, shared with 'edit' permission, or admin

**Request Body**:
```json
{
  "websdr_station_id": "websdr-001"
}
```

**Response**: `201 Created`
```json
{
  "id": "member-001",
  "constellation_id": "550e8400-e29b-41d4-a716-446655440000",
  "websdr_station_id": "websdr-001",
  "added_at": "2025-01-15T11:00:00Z",
  "added_by": "user-123"
}
```

**Error Responses**:
- `403 Forbidden`: No edit permission
- `404 Not Found`: Constellation or WebSDR does not exist
- `409 Conflict`: WebSDR already in constellation

#### Remove WebSDR from Constellation

```http
DELETE /api/v1/constellations/{constellation_id}/members/{websdr_station_id}
```

**Requirements**: Owner, shared with 'edit' permission, or admin

**Response**: `204 No Content`

**Error Responses**:
- `403 Forbidden`: No edit permission
- `404 Not Found`: Constellation, WebSDR, or membership does not exist

### Sharing Endpoints

#### List Shares for a Constellation

```http
GET /api/v1/constellations/{constellation_id}/shares
```

**Requirements**: Owner or admin only

**Response**: `200 OK`
```json
[
  {
    "id": "share-001",
    "constellation_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "user-456",
    "permission": "edit",
    "shared_by": "user-123",
    "shared_at": "2025-01-15T12:00:00Z",
    "user_info": {
      "email": "alice@example.com",
      "username": "alice"
    }
  }
]
```

**Error Responses**:
- `403 Forbidden`: Not owner or admin
- `404 Not Found`: Constellation does not exist

#### Create Share

```http
POST /api/v1/constellations/{constellation_id}/shares
```

**Requirements**: Owner or admin only

**Request Body**:
```json
{
  "user_id": "user-456",
  "permission": "edit"  // or "read"
}
```

**Response**: `201 Created`
```json
{
  "id": "share-001",
  "constellation_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user-456",
  "permission": "edit",
  "shared_by": "user-123",
  "shared_at": "2025-01-15T12:00:00Z"
}
```

**Error Responses**:
- `403 Forbidden`: Not owner or admin
- `404 Not Found`: Constellation does not exist
- `409 Conflict`: Share already exists for this user
- `422 Unprocessable Entity`: Invalid permission level

#### Update Share Permission

```http
PUT /api/v1/constellations/{constellation_id}/shares/{user_id}
```

**Requirements**: Owner or admin only

**Request Body**:
```json
{
  "permission": "read"
}
```

**Response**: `200 OK` (updated share object)

**Error Responses**:
- `403 Forbidden`: Not owner or admin
- `404 Not Found`: Constellation or share does not exist
- `422 Unprocessable Entity`: Invalid permission level

#### Delete Share

```http
DELETE /api/v1/constellations/{constellation_id}/shares/{user_id}
```

**Requirements**: Owner or admin only

**Response**: `204 No Content`

**Error Responses**:
- `403 Forbidden`: Not owner or admin
- `404 Not Found`: Constellation or share does not exist

---

## Best Practices

### For Operators

1. **Descriptive Names**: Use clear, descriptive names for constellations
   - ✅ "Northern Italy VHF Coverage"
   - ❌ "My Constellation 1"

2. **Share Appropriately**: 
   - Use 'Read' for most users (prevents accidental changes)
   - Use 'Edit' only for trusted collaborators who need to manage stations

3. **Document Your Resources**:
   - Use description fields to explain purpose and usage
   - Include contact info or team name

4. **Regular Cleanup**:
   - Delete unused constellations to keep the system organized
   - Remove outdated sources and models

### For Admins

1. **Monitor Ownership**: Regularly review resource ownership
2. **Backup Before Deletion**: Verify impact before deleting resources
3. **User Training**: Ensure users understand the RBAC system
4. **Access Audits**: Periodically review shares and permissions

### For Users

1. **Request Access**: Don't hesitate to request access to constellations you need
2. **Report Issues**: If permissions seem incorrect, contact your admin
3. **Respect Ownership**: Don't try to work around access restrictions

---

## FAQ

**Q: Can I transfer ownership of a constellation?**  
A: Currently, no. Only admins can delete and recreate resources to change ownership.

**Q: Can a constellation have multiple owners?**  
A: No. Each constellation has one owner. However, you can share with 'edit' permission to grant similar privileges.

**Q: What happens if a constellation owner leaves the organization?**  
A: Admins should reassign resources by recreating them or manually updating the database.

**Q: Can I see who shared a constellation with me?**  
A: Yes. The constellation details show the owner, and shares show `shared_by` field in API responses.

**Q: Can I share a constellation that was shared with me?**  
A: No. Only the owner can share resources.

**Q: Are WebSDR stations owned by anyone?**  
A: No. WebSDR stations are globally visible to all authenticated users. Only constellation assignments are controlled.

**Q: Can I use multiple constellations in a single localization session?**  
A: No. Each session is tied to one constellation. Create a constellation with all needed stations.

**Q: What's the difference between deleting a constellation and removing all its WebSDRs?**  
A: Removing WebSDRs keeps the constellation (you can re-add stations later). Deleting removes the constellation entirely.

---

## Support

If you encounter issues or have questions:

1. Check this guide and [FAQ](#faq)
2. Check the [Troubleshooting](#troubleshooting) section
3. Contact your system administrator
4. Report bugs at: `https://github.com/fulgidus/heimdall/issues`

---

**Last Updated**: 2025-11-08  
**Document Version**: 1.0  
**Contact**: alessio.corsi@gmail.com

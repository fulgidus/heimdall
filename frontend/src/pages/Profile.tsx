import React, { useState } from 'react';
import { useAuthStore } from '../store';

const Profile: React.FC = () => {
    const { user } = useAuthStore();
    const [activeTab, setActiveTab] = useState<'profile' | 'security' | 'activity'>('profile');
    const [isSaving, setIsSaving] = useState(false);

    // Profile state (TODO: Connect to backend user API)
    const [profile, setProfile] = useState({
        firstName: 'Admin',
        lastName: 'User',
        email: user?.email || 'admin@heimdall.local',
        phone: '+39 123 456 7890',
        organization: 'Heimdall SDR',
        role: 'System Administrator',
        location: 'Turin, Italy',
        bio: 'RF localization specialist',
    });

    const [security, setSecurity] = useState({
        currentPassword: '',
        newPassword: '',
        confirmPassword: '',
        twoFactorEnabled: false,
    });

    // Mock activity data
    const recentActivity = [
        {
            id: 1,
            action: 'Logged in',
            timestamp: new Date().toISOString(),
            ip: '192.168.1.100',
            device: 'Chrome on Windows',
        },
        {
            id: 2,
            action: 'Updated settings',
            timestamp: new Date(Date.now() - 3600000).toISOString(),
            ip: '192.168.1.100',
            device: 'Chrome on Windows',
        },
        {
            id: 3,
            action: 'Created recording session',
            timestamp: new Date(Date.now() - 7200000).toISOString(),
            ip: '192.168.1.100',
            device: 'Chrome on Windows',
        },
    ];

    const handleSaveProfile = async () => {
        setIsSaving(true);
        // TODO: Save to backend API
        await new Promise(resolve => setTimeout(resolve, 1000));
        setIsSaving(false);
    };

    const handleChangePassword = async () => {
        setIsSaving(true);
        // TODO: Change password via API
        await new Promise(resolve => setTimeout(resolve, 1000));
        setSecurity({ ...security, currentPassword: '', newPassword: '', confirmPassword: '' });
        setIsSaving(false);
    };

    return (
        <>
            {/* Breadcrumb */}
            <div className="page-header">
                <div className="page-block">
                    <div className="row align-items-center">
                        <div className="col-md-12">
                            <ul className="breadcrumb">
                                <li className="breadcrumb-item"><a href="/dashboard">Home</a></li>
                                <li className="breadcrumb-item"><a href="#">Settings</a></li>
                                <li className="breadcrumb-item" aria-current="page">Profile</li>
                            </ul>
                        </div>
                        <div className="col-md-12">
                            <div className="page-header-title">
                                <h2 className="mb-0">User Profile</h2>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="row">
                {/* Profile Sidebar */}
                <div className="col-lg-3">
                    <div className="card">
                        <div className="card-body text-center">
                            <div className="avtar avtar-xl bg-primary mx-auto mb-3">
                                <i className="ph ph-user f-40"></i>
                            </div>
                            <h5 className="mb-1">{profile.firstName} {profile.lastName}</h5>
                            <p className="text-muted f-12 mb-3">{profile.role}</p>
                            <div className="d-grid">
                                <button className="btn btn-outline-primary btn-sm mb-2">
                                    <i className="ph ph-image me-1"></i>
                                    Change Avatar
                                </button>
                            </div>
                        </div>
                    </div>

                    <div className="card">
                        <div className="card-body">
                            <div className="list-group list-group-flush">
                                <a
                                    href="#!"
                                    className={`list-group-item list-group-item-action ${activeTab === 'profile' ? 'active' : ''}`}
                                    onClick={(e) => {
                                        e.preventDefault();
                                        setActiveTab('profile');
                                    }}
                                >
                                    <i className="ph ph-user me-2"></i>
                                    Profile Info
                                </a>
                                <a
                                    href="#!"
                                    className={`list-group-item list-group-item-action ${activeTab === 'security' ? 'active' : ''}`}
                                    onClick={(e) => {
                                        e.preventDefault();
                                        setActiveTab('security');
                                    }}
                                >
                                    <i className="ph ph-shield me-2"></i>
                                    Security
                                </a>
                                <a
                                    href="#!"
                                    className={`list-group-item list-group-item-action ${activeTab === 'activity' ? 'active' : ''}`}
                                    onClick={(e) => {
                                        e.preventDefault();
                                        setActiveTab('activity');
                                    }}
                                >
                                    <i className="ph ph-activity me-2"></i>
                                    Activity Log
                                </a>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Profile Content */}
                <div className="col-lg-9">
                    <div className="card">
                        <div className="card-header">
                            <h5 className="mb-0">
                                {activeTab === 'profile' && 'Profile Information'}
                                {activeTab === 'security' && 'Security Settings'}
                                {activeTab === 'activity' && 'Recent Activity'}
                            </h5>
                        </div>
                        <div className="card-body">
                            {/* Profile Tab */}
                            {activeTab === 'profile' && (
                                <div className="row g-3">
                                    <div className="col-md-6">
                                        <label className="form-label">First Name</label>
                                        <input
                                            type="text"
                                            className="form-control"
                                            value={profile.firstName}
                                            onChange={(e) => setProfile({ ...profile, firstName: e.target.value })}
                                        />
                                    </div>
                                    <div className="col-md-6">
                                        <label className="form-label">Last Name</label>
                                        <input
                                            type="text"
                                            className="form-control"
                                            value={profile.lastName}
                                            onChange={(e) => setProfile({ ...profile, lastName: e.target.value })}
                                        />
                                    </div>
                                    <div className="col-md-6">
                                        <label className="form-label">Email</label>
                                        <input
                                            type="email"
                                            className="form-control"
                                            value={profile.email}
                                            onChange={(e) => setProfile({ ...profile, email: e.target.value })}
                                        />
                                    </div>
                                    <div className="col-md-6">
                                        <label className="form-label">Phone</label>
                                        <input
                                            type="tel"
                                            className="form-control"
                                            value={profile.phone}
                                            onChange={(e) => setProfile({ ...profile, phone: e.target.value })}
                                        />
                                    </div>
                                    <div className="col-md-6">
                                        <label className="form-label">Organization</label>
                                        <input
                                            type="text"
                                            className="form-control"
                                            value={profile.organization}
                                            onChange={(e) => setProfile({ ...profile, organization: e.target.value })}
                                        />
                                    </div>
                                    <div className="col-md-6">
                                        <label className="form-label">Role</label>
                                        <input
                                            type="text"
                                            className="form-control"
                                            value={profile.role}
                                            disabled
                                        />
                                    </div>
                                    <div className="col-12">
                                        <label className="form-label">Location</label>
                                        <input
                                            type="text"
                                            className="form-control"
                                            value={profile.location}
                                            onChange={(e) => setProfile({ ...profile, location: e.target.value })}
                                        />
                                    </div>
                                    <div className="col-12">
                                        <label className="form-label">Bio</label>
                                        <textarea
                                            className="form-control"
                                            rows={3}
                                            value={profile.bio}
                                            onChange={(e) => setProfile({ ...profile, bio: e.target.value })}
                                        ></textarea>
                                    </div>
                                    <div className="col-12">
                                        <button
                                            className="btn btn-primary"
                                            onClick={handleSaveProfile}
                                            disabled={isSaving}
                                        >
                                            {isSaving ? (
                                                <>
                                                    <span className="spinner-border spinner-border-sm me-2"></span>
                                                    Saving...
                                                </>
                                            ) : (
                                                <>
                                                    <i className="ph ph-check me-2"></i>
                                                    Save Changes
                                                </>
                                            )}
                                        </button>
                                    </div>
                                </div>
                            )}

                            {/* Security Tab */}
                            {activeTab === 'security' && (
                                <div className="row g-3">
                                    <div className="col-12">
                                        <h6>Change Password</h6>
                                    </div>
                                    <div className="col-12">
                                        <label className="form-label">Current Password</label>
                                        <input
                                            type="password"
                                            className="form-control"
                                            value={security.currentPassword}
                                            onChange={(e) => setSecurity({ ...security, currentPassword: e.target.value })}
                                        />
                                    </div>
                                    <div className="col-md-6">
                                        <label className="form-label">New Password</label>
                                        <input
                                            type="password"
                                            className="form-control"
                                            value={security.newPassword}
                                            onChange={(e) => setSecurity({ ...security, newPassword: e.target.value })}
                                        />
                                    </div>
                                    <div className="col-md-6">
                                        <label className="form-label">Confirm New Password</label>
                                        <input
                                            type="password"
                                            className="form-control"
                                            value={security.confirmPassword}
                                            onChange={(e) => setSecurity({ ...security, confirmPassword: e.target.value })}
                                        />
                                    </div>
                                    <div className="col-12">
                                        <button
                                            className="btn btn-primary"
                                            onClick={handleChangePassword}
                                            disabled={isSaving}
                                        >
                                            {isSaving ? (
                                                <>
                                                    <span className="spinner-border spinner-border-sm me-2"></span>
                                                    Updating...
                                                </>
                                            ) : (
                                                <>
                                                    <i className="ph ph-lock-key me-2"></i>
                                                    Update Password
                                                </>
                                            )}
                                        </button>
                                    </div>
                                    <div className="col-12">
                                        <hr />
                                        <h6>Two-Factor Authentication</h6>
                                        <div className="form-check form-switch mt-3">
                                            <input
                                                className="form-check-input"
                                                type="checkbox"
                                                id="twoFactor"
                                                checked={security.twoFactorEnabled}
                                                onChange={(e) => setSecurity({ ...security, twoFactorEnabled: e.target.checked })}
                                            />
                                            <label className="form-check-label" htmlFor="twoFactor">
                                                Enable Two-Factor Authentication
                                            </label>
                                        </div>
                                        <small className="text-muted">Add an extra layer of security to your account</small>
                                    </div>
                                </div>
                            )}

                            {/* Activity Tab */}
                            {activeTab === 'activity' && (
                                <div className="table-responsive">
                                    <table className="table table-hover mb-0">
                                        <thead>
                                            <tr>
                                                <th>Action</th>
                                                <th>Timestamp</th>
                                                <th>IP Address</th>
                                                <th>Device</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {recentActivity.map((activity) => (
                                                <tr key={activity.id}>
                                                    <td>
                                                        <div className="d-flex align-items-center">
                                                            <div className="avtar avtar-xs bg-light-primary">
                                                                <i className="ph ph-activity"></i>
                                                            </div>
                                                            <span className="ms-2">{activity.action}</span>
                                                        </div>
                                                    </td>
                                                    <td>{new Date(activity.timestamp).toLocaleString()}</td>
                                                    <td>{activity.ip}</td>
                                                    <td>{activity.device}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
};

export default Profile;

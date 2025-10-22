import React from 'react';
import { Card, MainLayout, Button, Input } from '../components';
import { Bell, Lock, User, Database, Shield } from 'lucide-react';

interface SettingItemProps {
    icon: React.ReactNode;
    title: string;
    description: string;
    action?: string;
}

const SettingItem: React.FC<SettingItemProps> = ({ icon, title, description, action }) => (
    <div className="flex items-center justify-between p-4 rounded-lg bg-sea-green bg-opacity-5 border border-neon-blue border-opacity-20 hover:bg-opacity-10 transition-all duration-200">
        <div className="flex items-center gap-4">
            <div className="text-neon-blue text-2xl">{icon}</div>
            <div>
                <h4 className="font-semibold text-white">{title}</h4>
                <p className="text-sm text-french-gray">{description}</p>
            </div>
        </div>
        {action && (
            <Button variant="accent" size="sm">
                {action}
            </Button>
        )}
    </div>
);

const Settings: React.FC = () => {
    return (
        <MainLayout title="Settings">
            <div className="space-y-8 max-w-4xl">
                {/* Profile Settings */}
                <section>
                    <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                        <User className="text-neon-blue" size={28} />
                        Profile Settings
                    </h2>
                    <Card variant="elevated">
                        <div className="space-y-4">
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                <Input label="First Name" placeholder="John" />
                                <Input label="Last Name" placeholder="Doe" />
                            </div>
                            <Input label="Email Address" type="email" placeholder="john@example.com" />
                            <Input label="Bio" placeholder="Tell us about yourself" />
                            <div className="flex gap-3 pt-4">
                                <Button variant="accent">Save Changes</Button>
                                <Button variant="secondary">Cancel</Button>
                            </div>
                        </div>
                    </Card>
                </section>

                {/* Security */}
                <section>
                    <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                        <Lock className="text-neon-blue" size={28} />
                        Security
                    </h2>
                    <div className="space-y-4">
                        <SettingItem
                            icon={<Shield />}
                            title="Two-Factor Authentication"
                            description="Add an extra layer of security to your account"
                            action="Enable"
                        />
                        <SettingItem
                            icon={<Lock />}
                            title="Change Password"
                            description="Update your password regularly for better security"
                            action="Change"
                        />
                        <Card variant="elevated" className="bg-opacity-5">
                            <div className="flex items-start gap-4">
                                <span className="text-2xl">⚠️</span>
                                <div>
                                    <h4 className="font-semibold text-white mb-1">Session Management</h4>
                                    <p className="text-sm text-french-gray mb-3">You have 3 active sessions</p>
                                    <Button variant="danger" size="sm">
                                        Sign Out All Devices
                                    </Button>
                                </div>
                            </div>
                        </Card>
                    </div>
                </section>

                {/* Notifications */}
                <section>
                    <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                        <Bell className="text-neon-blue" size={28} />
                        Notifications
                    </h2>
                    <Card variant="elevated">
                        <div className="space-y-4">
                            {[
                                { label: 'Email Notifications', description: 'Receive updates via email' },
                                { label: 'Push Notifications', description: 'Get push alerts on your device' },
                                { label: 'Weekly Digest', description: 'Receive weekly summary reports' },
                                { label: 'Marketing Emails', description: 'News and feature updates' },
                            ].map((item, idx) => (
                                <label key={idx} className="flex items-center gap-3 p-3 rounded-lg hover:bg-sea-green hover:bg-opacity-10 cursor-pointer transition-all duration-200">
                                    <input
                                        type="checkbox"
                                        defaultChecked={idx < 2}
                                        className="w-4 h-4 rounded border-neon-blue cursor-pointer accent-neon-blue"
                                    />
                                    <div className="flex-1">
                                        <p className="font-medium text-white">{item.label}</p>
                                        <p className="text-sm text-french-gray">{item.description}</p>
                                    </div>
                                </label>
                            ))}
                        </div>
                    </Card>
                </section>

                {/* Data Management */}
                <section>
                    <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                        <Database className="text-neon-blue" size={28} />
                        Data Management
                    </h2>
                    <div className="space-y-4">
                        <SettingItem
                            icon={<Database />}
                            title="Export Data"
                            description="Download all your data in CSV format"
                            action="Export"
                        />
                        <SettingItem
                            icon={<Database />}
                            title="Delete Account"
                            description="Permanently delete your account and all data"
                            action="Delete"
                        />
                    </div>
                </section>
            </div>
        </MainLayout>
    );
};

export default Settings;

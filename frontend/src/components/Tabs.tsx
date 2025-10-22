import React from 'react';
import classNames from 'classnames';

interface TabsProps {
    tabs: Array<{
        id: string;
        label: string;
        content: React.ReactNode;
        icon?: React.ReactNode;
    }>;
    defaultTabId?: string;
    onChange?: (tabId: string) => void;
}

const Tabs: React.FC<TabsProps> = ({ tabs, defaultTabId, onChange }) => {
    const [activeTab, setActiveTab] = React.useState(defaultTabId || tabs[0]?.id);

    const handleTabChange = (tabId: string) => {
        setActiveTab(tabId);
        onChange?.(tabId);
    };

    const activeTabContent = tabs.find((tab) => tab.id === activeTab);

    return (
        <div className="w-full">
            {/* Tab Headers */}
            <div className="flex gap-2 mb-6 border-b border-neon-blue border-opacity-20 overflow-x-auto">
                {tabs.map((tab) => (
                    <button
                        key={tab.id}
                        onClick={() => handleTabChange(tab.id)}
                        className={classNames(
                            'px-4 py-3 text-sm font-medium whitespace-nowrap transition-all duration-200 border-b-2 flex items-center gap-2',
                            activeTab === tab.id
                                ? 'text-neon-blue border-neon-blue'
                                : 'text-french-gray border-transparent hover:text-light-green'
                        )}
                    >
                        {tab.icon && <span>{tab.icon}</span>}
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Tab Content */}
            <div className="animate-fadeIn">
                {activeTabContent?.content}
            </div>
        </div>
    );
};

export default Tabs;

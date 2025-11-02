import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuthStore } from '../../store';

interface MenuItem {
  label: string;
  path: string;
  icon: string;
}

interface MenuSection {
  caption: string;
  icon?: string;
  items: MenuItem[];
}

interface DattaLayoutProps {
  children: React.ReactNode;
}

const DattaLayout: React.FC<DattaLayoutProps> = ({ children }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuthStore();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const menuSections: MenuSection[] = [
    {
      caption: 'Navigation',
      items: [{ label: 'Dashboard', path: '/dashboard', icon: 'ph-house-line' }],
    },
    {
      caption: 'RF Operations',
      icon: 'ph-radio',
      items: [
        { label: 'Localization', path: '/localization', icon: 'ph-map-pin' },
        { label: 'Sources Management', path: '/sources', icon: 'ph-broadcast' },
        { label: 'WebSDR Management', path: '/websdrs', icon: 'ph-radio-button' },

        { label: 'Recording Session', path: '/recording', icon: 'ph-record' },
        { label: 'Session History', path: '/history', icon: 'ph-clock-clockwise' },
      ],
    },
    {
      caption: 'Analysis',
      icon: 'ph-chart-line',
      items: [
        { label: 'Analytics', path: '/analytics', icon: 'ph-chart-bar' },
        { label: 'System Status', path: '/system-status', icon: 'ph-activity' },
      ],
    },
    {
      caption: 'ML & Training',
      icon: 'ph-brain',
      items: [
        { label: 'Training Dashboard', path: '/training', icon: 'ph-graduation-cap' },
      ],
    },
    {
      caption: 'Settings',
      icon: 'ph-gear',
      items: [
        { label: 'Settings', path: '/settings', icon: 'ph-gear-six' },
        { label: 'Profile', path: '/profile', icon: 'ph-user-circle' },
        { label: 'Import/Export', path: '/import-export', icon: 'ph-download' },
      ],
    },
  ];

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const isActivePath = (path: string) => {
    return location.pathname === path;
  };

  return (
    <>
      {/* Loader */}
      <div className="loader-bg" style={{ display: 'none' }}>
        <div className="loader-track">
          <div className="loader-fill"></div>
        </div>
      </div>

      {/* Sidebar */}
      <nav className={`pc-sidebar ${sidebarCollapsed ? 'mob-sidebar-active' : ''}`}>
        <div className="navbar-wrapper">
          <div className="m-header">
            <Link to="/dashboard" className="b-brand text-primary">
              <span className="badge bg-brand-color-2 rounded-pill ms-2 theme-version">v1.0</span>
              <span className="text-white fw-bold fs-4 ms-2">Heimdall</span>
            </Link>
          </div>
          <div className="navbar-content">
            <ul className="pc-navbar">
              {menuSections.map((section, idx) => (
                <React.Fragment key={idx}>
                  <li className="pc-item pc-caption">
                    <label>{section.caption}</label>
                    {section.icon && <i className={`ph ${section.icon}`}></i>}
                  </li>
                  {section.items.map((item, itemIdx) => (
                    <li
                      key={itemIdx}
                      className={`pc-item ${isActivePath(item.path) ? 'active' : ''}`}
                    >
                      <Link to={item.path} className="pc-link">
                        <span className="pc-micon">
                          <i className={`ph ${item.icon}`}></i>
                        </span>
                        <span className="pc-mtext">{item.label}</span>
                      </Link>
                    </li>
                  ))}
                </React.Fragment>
              ))}
            </ul>

            {/* User Card */}
            <div className="card pc-user-card my-3">
              <div className="card-body">
                <div className="d-flex align-items-center">
                  <div className="flex-shrink-0">
                    <div className="avtar avtar-s bg-primary">
                      <i className="ph ph-user"></i>
                    </div>
                  </div>
                  <div className="flex-grow-1 ms-3">
                    <div className="dropdown">
                      <a
                        href="#"
                        className="text-decoration-none text-white dropdown-toggle"
                        data-bs-toggle="dropdown"
                        aria-expanded="false"
                      >
                        <div className="d-inline-block">
                          <h6 className="mb-0 text-truncate" style={{ maxWidth: '120px' }}>
                            {user?.email?.split('@')[0] || 'User'}
                          </h6>
                          <small className="text-muted">Operator</small>
                        </div>
                      </a>
                      <ul className="dropdown-menu">
                        <li>
                          <Link className="dropdown-item" to="/profile">
                            <i className="ph ph-user-circle me-2"></i>
                            Profile
                          </Link>
                        </li>
                        <li>
                          <Link className="dropdown-item" to="/settings">
                            <i className="ph ph-gear-six me-2"></i>
                            Settings
                          </Link>
                        </li>
                        <li>
                          <hr className="dropdown-divider" />
                        </li>
                        <li>
                          <a className="dropdown-item" href="#" onClick={handleLogout}>
                            <i className="ph ph-sign-out me-2"></i>
                            Logout
                          </a>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Header */}
      <header className="pc-header">
        <div className="header-wrapper">
          <div className="me-auto pc-mob-drp">
            <ul className="list-unstyled">
              <li className="pc-h-item pc-sidebar-collapse">
                <a
                  href="#"
                  className="pc-head-link ms-0"
                  id="sidebar-hide"
                  onClick={e => {
                    e.preventDefault();
                    document.body.classList.toggle('pc-sidebar-hide');
                  }}
                >
                  <i className="ti ti-menu-2"></i>
                </a>
              </li>
              <li className="pc-h-item pc-sidebar-popup">
                <a
                  href="#"
                  className="pc-head-link ms-0"
                  id="mobile-collapse"
                  onClick={e => {
                    e.preventDefault();
                    setSidebarCollapsed(!sidebarCollapsed);
                  }}
                >
                  <i className="ti ti-menu-2"></i>
                </a>
              </li>
            </ul>
          </div>
          <div className="ms-auto">
            <ul className="list-unstyled">
              <li className="dropdown pc-h-item">
                <a
                  className="pc-head-link dropdown-toggle arrow-none me-0"
                  data-bs-toggle="dropdown"
                  href="#"
                  role="button"
                  aria-haspopup="false"
                  aria-expanded="false"
                >
                  <i className="ph ph-bell"></i>
                  <span className="badge bg-success pc-h-badge">3</span>
                </a>
                <div className="dropdown-menu dropdown-menu-end pc-h-dropdown">
                  <div className="dropdown-header d-flex align-items-center justify-content-between">
                    <h5 className="m-0">Notifications</h5>
                    <a href="#!" className="btn btn-link btn-sm">
                      Mark all read
                    </a>
                  </div>
                  <div
                    className="dropdown-body text-wrap header-notification-scroll position-relative"
                    style={{ maxHeight: '250px' }}
                  >
                    <p className="text-span">Today</p>
                    <div className="card mb-2">
                      <div className="card-body">
                        <div className="d-flex">
                          <div className="flex-shrink-0">
                            <i className="ph ph-radio text-success"></i>
                          </div>
                          <div className="flex-grow-1 ms-3">
                            <span className="float-end text-muted">2 min ago</span>
                            <h6 className="text-body mb-2">WebSDR Online</h6>
                            <p className="mb-0">All receivers operational</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="text-center py-2">
                    <a href="#!" className="link-danger">
                      Clear all Notifications
                    </a>
                  </div>
                </div>
              </li>
              <li className="dropdown pc-h-item header-user-profile">
                <a
                  className="pc-head-link dropdown-toggle arrow-none me-0"
                  data-bs-toggle="dropdown"
                  href="#"
                  role="button"
                  aria-haspopup="false"
                  data-bs-auto-close="outside"
                  aria-expanded="false"
                >
                  <div className="avtar avtar-s bg-primary">
                    <i className="ph ph-user"></i>
                  </div>
                </a>
                <div className="dropdown-menu dropdown-menu-end pc-h-dropdown">
                  <div className="dropdown-header d-flex align-items-center justify-content-between">
                    <h5 className="m-0">Profile</h5>
                  </div>
                  <div className="dropdown-body">
                    <div
                      className="profile-notification-scroll position-relative"
                      style={{ maxHeight: '250px' }}
                    >
                      <div className="d-flex mb-3">
                        <div className="flex-shrink-0">
                          <div className="avtar avtar-m bg-primary">
                            <i className="ph ph-user"></i>
                          </div>
                        </div>
                        <div className="flex-grow-1 ms-3">
                          <h6 className="mb-1">{user?.email?.split('@')[0] || 'User'}</h6>
                          <span>{user?.email || 'user@heimdall.sdr'}</span>
                        </div>
                      </div>
                      <hr className="border-secondary border-opacity-50" />
                      <Link to="/profile" className="dropdown-item">
                        <i className="ph ph-user-circle"></i>
                        <span>Edit Profile</span>
                      </Link>
                      <Link to="/settings" className="dropdown-item">
                        <i className="ph ph-gear-six"></i>
                        <span>Settings</span>
                      </Link>
                      <hr className="border-secondary border-opacity-50" />
                      <a href="#" onClick={handleLogout} className="dropdown-item">
                        <i className="ph ph-sign-out"></i>
                        <span>Logout</span>
                      </a>
                    </div>
                  </div>
                </div>
              </li>
            </ul>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="pc-container">
        <div className="pc-content">{children}</div>
      </div>

      {/* Mobile Overlay */}
      {sidebarCollapsed && (
        <div
          className="pc-sidebar-hide"
          onClick={() => setSidebarCollapsed(false)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.5)',
            zIndex: 1029,
          }}
        />
      )}
    </>
  );
};

export default DattaLayout;

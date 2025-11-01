import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import Login from './Login';

// Mock useNavigate
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => vi.fn(),
  };
});

function renderLogin() {
  return render(
    <BrowserRouter>
      <Login />
    </BrowserRouter>
  );
}

describe('Login Component', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  describe('Rendering', () => {
    it('should render login form with all required fields', () => {
      renderLogin();

      // Verify form is present by checking key elements
      expect(screen.getByPlaceholderText('admin@heimdall.local')).toBeInTheDocument();
      expect(screen.getByPlaceholderText(/••••/)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Sign In/i })).toBeInTheDocument();
    });

    it('should display Heimdall logo and platform name', () => {
      renderLogin();

      // Find by role since getByText finds multiple matches
      const heading = screen.getByRole('heading', { level: 1 });
      expect(heading).toHaveTextContent('Heimdall');

      // Check for platform name in subtitle
      const subtitle = screen.getByText(/RF Source Localization Platform/i);
      expect(subtitle).toBeInTheDocument();
    });

    it('should display demo credentials info', () => {
      renderLogin();

      expect(screen.getByText(/Demo Credentials/i)).toBeInTheDocument();
      expect(screen.getByText(/admin@heimdall.local/)).toBeInTheDocument();
    });
  });

  describe('Password Visibility Toggle', () => {
    it('should toggle password visibility', () => {
      renderLogin();

      const passwordInput = screen.getByPlaceholderText(/••••/) as HTMLInputElement;
      const toggleButton = screen.getByLabelText(/Show password/i);

      expect(passwordInput.type).toBe('password');

      fireEvent.click(toggleButton);
      expect(passwordInput.type).toBe('text');

      fireEvent.click(toggleButton);
      expect(passwordInput.type).toBe('password');
    });
  });

  describe('Form Validation', () => {
    it('should require email field', () => {
      renderLogin();

      const emailInput = screen.getByPlaceholderText('admin@heimdall.local') as HTMLInputElement;
      expect(emailInput.required).toBe(true);
    });

    it('should require password field', () => {
      renderLogin();

      const passwordInput = screen.getByPlaceholderText(/••••/) as HTMLInputElement;
      expect(passwordInput.required).toBe(true);
    });

    it('should have correct input types', () => {
      renderLogin();

      const emailInput = screen.getByPlaceholderText('admin@heimdall.local') as HTMLInputElement;
      const passwordInput = screen.getByPlaceholderText(/••••/) as HTMLInputElement;

      expect(emailInput.type).toBe('email');
      expect(passwordInput.type).toBe('password');
    });
  });

  describe('Form Input Handling', () => {
    it('should update email input value', () => {
      renderLogin();

      const emailInput = screen.getByPlaceholderText('admin@heimdall.local') as HTMLInputElement;
      fireEvent.change(emailInput, { target: { value: 'newemail@test.com' } });

      expect(emailInput.value).toBe('newemail@test.com');
    });

    it('should update password input value', () => {
      renderLogin();

      const passwordInput = screen.getByPlaceholderText(/••••/) as HTMLInputElement;
      fireEvent.change(passwordInput, { target: { value: 'MyPassword123' } });

      expect(passwordInput.value).toBe('MyPassword123');
    });

    it('should have correct placeholder texts', () => {
      renderLogin();

      expect(screen.getByPlaceholderText('admin@heimdall.local')).toBeInTheDocument();
      expect(screen.getByPlaceholderText(/••••/)).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper labels for inputs', () => {
      renderLogin();

      expect(screen.getByText('Email Address')).toBeInTheDocument();
      expect(screen.getByText('Password')).toBeInTheDocument();
    });

    it('should have aria-label for show/hide password button', () => {
      renderLogin();

      expect(screen.getByLabelText(/Show password/i)).toBeInTheDocument();
    });
  });

  describe('UI Elements Styling', () => {
    it('should have correct CSS classes for email input', () => {
      renderLogin();

      const emailInput = screen.getByPlaceholderText('admin@heimdall.local');
      expect(emailInput).toHaveClass('login-form-input');
    });

    it('should have correct CSS classes for password input', () => {
      renderLogin();

      const passwordInput = screen.getByPlaceholderText(/••••/);
      expect(passwordInput).toHaveClass('login-form-input');
      expect(passwordInput).toHaveClass('login-password-input');
    });

    it('should have correct styling for submit button', () => {
      renderLogin();

      const submitButton = screen.getByRole('button', { name: /Sign In/i });
      expect(submitButton).toHaveClass('login-submit-button');
    });
  });
});

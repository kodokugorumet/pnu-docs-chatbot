import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    include: ['tests/frontend/**/*.test.{ts,tsx}'],
    setupFiles: ['tests/frontend/setup.ts'],
    restoreMocks: true,
    coverage: {
      provider: 'v8',
      include: [
        'src/App.tsx',
        'src/components/AnswerContent.tsx',
        'src/components/CitationLocation.tsx',
        'src/components/ChatComposer.tsx',
        'src/components/ChatSidebar.tsx',
        'src/components/Dialog.tsx',
        'src/hooks/useConversations.ts',
        'src/state/conversations.ts',
        'src/state/conversationStorage.ts',
        'src/chat/presentation.ts',
        'src/chat/settings.ts',
        'src/state/messageValidation.ts',
      ],
      reportsDirectory: 'coverage/frontend',
      reporter: ['text', 'json', 'json-summary', 'html'],
      thresholds: {
        lines: 100,
        branches: 100,
        functions: 100,
        statements: 100,
      },
    },
  },
})

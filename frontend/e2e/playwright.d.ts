declare namespace NodeJS {
    interface ProcessEnv {
        APP_USER_EMAIL?: string;
        APP_USER_PASSWORD?: string;
        TEST_BACKEND_ORIGIN?: string;
        CI?: string;
    }
}

declare const process: {
    env: NodeJS.ProcessEnv;
};

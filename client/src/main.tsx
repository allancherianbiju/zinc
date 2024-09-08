import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { NextUIProvider } from "@nextui-org/react";
import { ThemeProvider as NextThemesProvider } from "next-themes";
import { GoogleOAuthProvider } from "@react-oauth/google";
import { BrowserRouter as Router } from "react-router-dom";
import App from "./App.tsx";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <NextThemesProvider
      attribute="class"
      defaultTheme="system"
      themes={["light", "dark"]}
    >
      <NextUIProvider>
        <GoogleOAuthProvider clientId="94145857935-a553bvkq13c2o08gjgnic0d6og5lkdpp.apps.googleusercontent.com">
          {/* <Router> */}
          <App />
          {/* </Router> */}
        </GoogleOAuthProvider>
      </NextUIProvider>
    </NextThemesProvider>
  </StrictMode>
);

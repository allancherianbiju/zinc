import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  Card,
  CardBody,
  Progress,
  Button,
  ScrollShadow,
  CardHeader,
} from "@nextui-org/react";
import Navbar from "../components/navbar";
import api from "../api";

interface StatusMessage {
  message: string;
  progress: number;
  eta?: string;
  timestamp: string;
}

const Processing = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Initializing...");
  const [error, setError] = useState<string | null>(null);
  const [statusMessages, setStatusMessages] = useState<StatusMessage[]>([]);
  const [eta, setEta] = useState<string>("Calculating...");
  const { mapping, fileData } = location.state || {};

  useEffect(() => {
    const controller = new AbortController();

    const processData = async () => {
      try {
        const response = await api.post(
          "/process",
          { mapping: mapping },
          {
            responseType: "text",
            headers: {
              Accept: "text/event-stream",
              "Content-Type": "application/json",
            },
            signal: controller.signal,
            onDownloadProgress: (progressEvent) => {
              const data = (progressEvent.event.target as XMLHttpRequest)
                .response;
              const lines = data.split("\n").filter(Boolean);

              lines.forEach((line: string) => {
                try {
                  const data = JSON.parse(line);
                  if (data.error) {
                    setError(data.error);
                    setStatus("Error occurred");
                  } else {
                    setProgress(data.progress || 0);
                    setStatus(data.message || "Processing...");
                    if (data.eta) {
                      setEta(data.eta);
                    }

                    setStatusMessages((prev) => [
                      ...prev,
                      {
                        message: data.message || "Processing...",
                        progress: data.progress || 0,
                        eta: data.eta,
                        timestamp: new Date().toLocaleTimeString(),
                      },
                    ]);
                  }
                } catch (e) {
                  console.error("Failed to parse update:", e);
                }
              });
            },
          }
        );
      } catch (error: any) {
        if (error.name === "AbortError" || error.message === "canceled") {
          // Ignore abort errors
          return;
        }
        console.error("Error processing data:", error);
        setStatus("Error occurred while processing data");
        setError(error.response?.data?.detail || error.message);
        setProgress(100);
      }
    };

    processData();

    return () => {
      controller.abort();
    };
  }, [mapping]);

  return (
    <div className="flex flex-col min-h-screen bg-background">
      <Navbar />
      <main className="flex-grow container mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold mb-4">Processing Data</h1>
        <Card className="mb-6">
          <CardBody>
            <div className="flex justify-between items-center mb-2">
              <p className="font-semibold">{status}</p>
              <p className="text-sm text-gray-500">{eta}</p>
            </div>
            <Progress
              value={progress}
              className="mb-4"
              color={error ? "danger" : "primary"}
            />
            {error && (
              <p className="text-red-500 mt-2">Error details: {error}</p>
            )}
          </CardBody>
        </Card>

        {/* Status Messages Card */}
        <Card className="mb-6">
          <CardHeader>
            <h3 className="text-lg font-semibold">Processing Log</h3>
          </CardHeader>
          <CardBody>
            <ScrollShadow className="h-[300px]">
              {statusMessages.map((message, index) => (
                <div
                  key={index}
                  className="py-2 border-b border-gray-700 last:border-0"
                >
                  <span className="text-gray-500 text-sm">
                    {new Date().toLocaleTimeString()} -
                  </span>
                  <span className="ml-2">{message.message}</span>
                </div>
              ))}
            </ScrollShadow>
          </CardBody>
        </Card>

        {progress === 100 && !error && (
          <Button color="primary" onClick={() => navigate("/report")}>
            View Report
          </Button>
        )}
        {error && (
          <Button
            color="primary"
            onClick={() =>
              navigate("/mapping", {
                state: {
                  fileData,
                  mapping,
                },
              })
            }
          >
            Back to Mapping
          </Button>
        )}
      </main>
    </div>
  );
};

export default Processing;

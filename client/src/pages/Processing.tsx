import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  Card,
  CardBody,
  Progress,
  Button,
  ScrollShadow,
  CardHeader,
  Spinner,
} from "@nextui-org/react";
import Navbar from "../components/navbar";
import api from "../api";
import { IconProgressCheck } from "@tabler/icons-react";

interface StatusMessage {
  message: string;
  progress: number;
  eta?: string;
  timestamp: string;
  id?: string;
  completed?: boolean;
  final?: boolean;
}

const Processing = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Initializing...");
  const [error, setError] = useState<string | null>(null);
  const [statusMessages, setStatusMessages] = useState<StatusMessage[]>([]);
  const [eta, setEta] = useState<string>("Calculating...");
  const [initialEta, setInitialEta] = useState<number | null>(null);
  const [remainingSeconds, setRemainingSeconds] = useState<number | null>(null);
  const [countdownStarted, setCountdownStarted] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  const { mapping, fileData } = location.state || {};

  const formatTimeRemaining = (seconds: number): string => {
    if (!seconds || seconds <= 0) {
      return "Almost done...";
    }

    if (seconds < 60) {
      return `${Math.ceil(seconds)} seconds remaining`;
    }

    const minutes = Math.floor(seconds / 60);
    const remainingSecs = Math.ceil(seconds % 60);

    if (minutes === 0) {
      return `${remainingSecs} seconds remaining`;
    } else if (remainingSecs === 0) {
      return `${minutes} minute${minutes > 1 ? "s" : ""} remaining`;
    } else {
      return `${minutes} minute${
        minutes > 1 ? "s" : ""
      } ${remainingSecs} seconds remaining`;
    }
  };

  const isDuplicateMessage = (
    newMessage: StatusMessage,
    existingMessages: StatusMessage[]
  ): boolean => {
    return existingMessages.some((msg) => msg.message === newMessage.message);
  };

  // Function to start the countdown
  const startCountdown = (totalSeconds: number) => {
    console.log("Starting countdown with", totalSeconds, "seconds"); // Debug log
    setInitialEta(totalSeconds);
    setRemainingSeconds(totalSeconds);
    setCountdownStarted(true);
  };

  // Countdown effect
  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    if (
      countdownStarted &&
      remainingSeconds !== null &&
      remainingSeconds >= 0
    ) {
      intervalId = setInterval(() => {
        setRemainingSeconds((prev) => {
          if (prev === null || prev <= 0) {
            setEta("Almost done..."); // Set "Almost done..." when countdown reaches 0
            return 0;
          }
          const newValue = prev - 1;
          // Update ETA display with remaining time
          setEta(formatTimeRemaining(newValue));
          return newValue;
        });

        // Calculate progress based on remaining time
        if (initialEta !== null && remainingSeconds !== null) {
          const progressIncrement = 20; // Progress from 30% to 80%
          const timeProgress = 1 - remainingSeconds / initialEta;
          const newProgress = 30 + timeProgress * progressIncrement;
          setProgress(Math.min(80, newProgress)); // Cap at 80%
        }
      }, 1000);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [countdownStarted, remainingSeconds, initialEta]);

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
                    setProgress(100);
                  } else {
                    setStatus(data.message || "Processing...");

                    // Handle completion state
                    if (data.completed) {
                      setIsCompleted(true);
                      setProgress(100);
                      setEta(data.eta || "Done");
                      setCountdownStarted(false);
                    } else if (
                      typeof data.eta === "number" &&
                      !countdownStarted
                    ) {
                      startCountdown(data.eta);
                    }

                    // Only update progress if we're not completed and not in countdown mode
                    if (!data.completed && !countdownStarted) {
                      setProgress(data.progress || 0);
                    }

                    const newMessage = {
                      message: data.message || "Processing...",
                      progress: data.progress || 0,
                      eta: data.completed
                        ? "Done"
                        : countdownStarted
                        ? formatTimeRemaining(remainingSeconds || 0)
                        : "Calculating...",
                      timestamp: new Date().toLocaleTimeString(),
                      id: data.id || Date.now().toString(),
                      completed: data.completed,
                      final: data.final,
                    };

                    setStatusMessages((prev) => {
                      if (!isDuplicateMessage(newMessage, prev)) {
                        return [...prev, newMessage];
                      }
                      return prev;
                    });
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
        <div className="flex items-center gap-2">
          {error || isCompleted ? (
            ""
          ) : (
            <Spinner size="sm" className="mb-3 mr-2" />
          )}
          <h1 className="text-2xl font-bold mb-4">Processing Data</h1>
        </div>
        <Card className="mb-6">
          <CardBody>
            <div className="flex justify-between items-center mb-2">
              <p className="font-semibold">{status}</p>
              <p className="text-sm text-gray-500">{eta}</p>
            </div>
            <Progress
              value={progress}
              className="mb-4"
              color={error ? "danger" : "secondary"}
            />
            {error && (
              <p className="text-red-500 mt-2">Error details: {error}</p>
            )}
          </CardBody>
        </Card>

        {/* Status Messages Card */}
        <Card className="mb-6">
          <CardHeader>
            <div className="flex items-center gap-2">
              <IconProgressCheck className="text-primary" size={24} />
              <h2 className="text-xl font-semibold">Processing Log</h2>
            </div>
          </CardHeader>
          <CardBody>
            <ScrollShadow className="h-[300px]" hideScrollBar>
              {statusMessages.map((message, index) => (
                <div
                  key={index}
                  className="py-2 border-b border-gray-700 last:border-0"
                >
                  <span className="text-gray-500 text-sm">
                    {message.timestamp || new Date().toLocaleTimeString()} -
                  </span>
                  <span className="ml-2">{message.message}</span>
                </div>
              ))}
            </ScrollShadow>
          </CardBody>
        </Card>

        <div className="flex justify-between">
          <div>
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
          </div>
          <div>
            {isCompleted && !error && (
              <Button color="primary" onClick={() => navigate("/report")}>
                View Report
              </Button>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Processing;

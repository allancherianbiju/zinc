import React, { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import api from "../api";
import Navbar from "../components/navbar";
import {
  Card,
  CardBody,
  Button,
  ScrollShadow,
  Tooltip,
} from "@nextui-org/react";
import {
  IconBrandGithub,
  IconUser,
  IconLink,
  IconCheck,
  IconCopy,
  IconCircleCheckFilled,
  IconPlugConnected,
} from "@tabler/icons-react";

export const ScanPage = () => {
  const location = useLocation();
  const { scanId, verificationData } = location.state;
  const [scanResults, setScanResults] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isCopied, setIsCopied] = useState(false);

  useEffect(() => {
    const pollScanStatus = async () => {
      try {
        const response = await api.get(`/api/scan/status/${scanId}`);
        if (response.data.status === "complete") {
          setScanResults(response.data.results);
          setIsLoading(false);
          clearInterval(interval);
        }
      } catch (error) {
        console.error("Error polling scan status:", error);
      }
    };

    const interval = setInterval(pollScanStatus, 2000);
    return () => clearInterval(interval);
  }, [scanId]);

  const handleCopyContent = () => {
    if (scanResults) {
      navigator.clipboard.writeText(scanResults);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-background">
      <Navbar />
      <main className="flex-grow container mx-auto px-4 py-8">
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-2xl font-bold">Scan Results</h1>
          <Tooltip
            placement="bottom"
            content={
              <div className="px-1 py-2">
                <div className="font-bold">
                  Integration with GitHub Actions and CI/CD pipelines
                </div>
                <div className="text-small">(Coming Soon)</div>
              </div>
            }
          >
            <Button variant="light" color="warning" isIconOnly>
              <IconPlugConnected size={18} />
            </Button>
          </Tooltip>
        </div>

        {verificationData && (
          <Card className="mb-6">
            <CardBody>
              <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                <IconCircleCheckFilled className="text-success" />
                Repository Verification Results
              </h2>
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <IconBrandGithub className="text-default-500" size={20} />
                  <span className="font-semibold">Repository Name:</span>
                  <span className="text-default-500">
                    {verificationData.repository.name}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <IconUser className="text-default-500" size={20} />
                  <span className="font-semibold">Owner:</span>
                  <span className="text-default-500">
                    {verificationData.repository.owner}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <IconLink className="text-default-500" size={20} />
                  <span className="font-semibold">URL:</span>
                  <a
                    href={verificationData.repository.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline"
                  >
                    {verificationData.repository.url}
                  </a>
                </div>
              </div>
            </CardBody>
          </Card>
        )}

        <Card className="mb-6">
          <CardBody>
            {isLoading ? (
              <div className="text-center p-8">
                <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full mx-auto mb-4"></div>
                <p>Scanning repository...</p>
              </div>
            ) : (
              <div className="relative">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-bold">Analysis Results</h3>
                  <Button
                    size="sm"
                    variant="light"
                    isIconOnly
                    onClick={handleCopyContent}
                    className="absolute top-0 right-0"
                  >
                    {isCopied ? (
                      <IconCheck size={18} />
                    ) : (
                      <IconCopy size={18} />
                    )}
                  </Button>
                </div>
                <ScrollShadow className="max-h-[600px]" hideScrollBar>
                  <div className="markdown-container prose prose-sm max-w-none dark:prose-invert prose-headings:text-foreground prose-p:text-default-600 prose-strong:text-foreground prose-pre:bg-default-100 prose-pre:text-default-600 prose-code:text-default-600">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {scanResults || ""}
                    </ReactMarkdown>
                  </div>
                </ScrollShadow>
              </div>
            )}
          </CardBody>
        </Card>
      </main>
    </div>
  );
};

import React, { useState, useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  Card,
  CardBody,
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Button,
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  useDisclosure,
  Pagination,
  Divider,
} from "@nextui-org/react";
import Navbar from "../components/navbar";
import api from "../api";
import ReactMarkdown from "react-markdown";
import {
  IconCopy,
  IconCheck,
  IconBrain,
  IconCalendar,
  IconClock,
  IconInfoCircle,
  IconArrowsExchange2,
  IconDatabase,
  IconMoodHappy,
  IconArrowRight,
  IconBulb,
  IconTrendingUp,
  IconTrendingDown,
  IconAlertTriangle,
  IconArrowUpRight,
} from "@tabler/icons-react";
import { IncidentTimingChart } from "../components/IncidentTimingChart";
import { CustomerSatisfactionChart } from "../components/CustomerSatisfactionChart";
import Particles, { initParticlesEngine } from "@tsparticles/react";
import { loadSlim } from "@tsparticles/slim";
import type { Engine } from "@tsparticles/engine";

type TimeUnit = "minutes" | "hours" | "days" | "seconds";

interface ActionableInsight {
  impact: "high" | "medium" | "low";
  category: string;
  value: string | number;
  trend: "up" | "down";
  recommendation: string;
}

const Report = () => {
  const navigate = useNavigate();
  const [reportData, setReportData] = useState<any>(null);
  const [selectedRow, setSelectedRow] = useState<any>(null);
  const [sop, setSop] = useState<string>("");
  const [page, setPage] = useState(1);
  const rowsPerPage = 10;
  const [sortField, setSortField] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc");
  const [isGeneratingSOP, setIsGeneratingSOP] = useState(false);
  const [isCopied, setIsCopied] = useState(false);
  const [timeUnit, setTimeUnit] = useState<TimeUnit>("minutes");
  const [showPositiveGroups, setShowPositiveGroups] = useState(false);
  const [showLeastComplex, setShowLeastComplex] = useState(false);
  const [init, setInit] = useState(false);

  const { isOpen, onOpen, onClose } = useDisclosure();
  const {
    isOpen: isInfoModalOpen,
    onOpen: onInfoModalOpen,
    onClose: onInfoModalClose,
  } = useDisclosure();

  useEffect(() => {
    const fetchReportData = async () => {
      try {
        const user = JSON.parse(localStorage.getItem("user") || "{}");
        if (!user.email) {
          console.warn("No user email found in localStorage");
          navigate("/");
          return;
        }

        console.log("Fetching report data for user:", user.email);
        const response = await api.get(`/report/${user.email}`);
        console.log("Report data response:", response.data);

        if (!response.data) {
          console.warn("No report data received");
          navigate("/");
          return;
        }

        console.log("Actionable insights:", response.data.actionable_insights);
        if (
          !response.data.actionable_insights ||
          response.data.actionable_insights.length === 0
        ) {
          console.warn("No actionable insights found in report data");
        }

        setReportData(response.data);
      } catch (error) {
        console.error("Error fetching report data:", error);
        navigate("/");
      }
    };

    fetchReportData();
  }, [navigate]);

  useEffect(() => {
    initParticlesEngine(async (engine: Engine) => {
      await loadSlim(engine);
    }).then(() => {
      setInit(true);
    });
  }, []);

  const particlesOptions = useMemo(
    () => ({
      background: {
        color: {
          value: "transparent",
        },
      },
      fpsLimit: 60,
      particles: {
        color: {
          value: "#ffffff",
        },
        links: {
          color: "#ffffff",
          distance: 150,
          enable: true,
          opacity: 0.2,
          width: 1,
        },
        move: {
          enable: true,
          speed: 1,
        },
        number: {
          density: {
            enable: true,
            area: 800,
          },
          value: 50,
        },
        opacity: {
          value: 0.2,
        },
        size: {
          value: { min: 1, max: 3 },
        },
      },
      detectRetina: true,
    }),
    []
  );

  const handleOpenModal = (row: any) => {
    setSelectedRow(row);
    setSop("");
    onOpen();
  };

  const handleGenerateSOP = async () => {
    if (!selectedRow) return;
    setIsGeneratingSOP(true);
    try {
      const response = await api.post("/generate_sop", {
        issue_description: selectedRow.issue_description,
        resolution_notes: selectedRow.resolution_notes,
      });
      setSop(response.data.sop);
    } catch (error) {
      console.error("Error generating SOP:", error);
    } finally {
      setIsGeneratingSOP(false);
    }
  };

  const handleCopySOP = () => {
    navigator.clipboard.writeText(sop);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return "success";
    if (score >= 60) return "warning";
    return "danger";
  };

  const getScoreText = (score: number) => {
    if (score >= 80) return "Outstanding";
    if (score >= 60) return "Average";
    return "Needs Improvement";
  };

  const convertTime = (
    minutes: number,
    unit: TimeUnit
  ): { value: number; unit: TimeUnit } => {
    switch (unit) {
      case "seconds":
        return { value: minutes * 60, unit: "seconds" };
      case "hours":
        return { value: minutes / 60, unit: "hours" };
      case "days":
        return { value: minutes / (60 * 24), unit: "days" };
      default:
        return { value: minutes, unit: "minutes" };
    }
  };

  const handleTimeUnitChange = () => {
    const units: TimeUnit[] = ["minutes", "hours", "days", "seconds"];
    const currentIndex = units.indexOf(timeUnit);
    const nextIndex = (currentIndex + 1) % units.length;
    setTimeUnit(units[nextIndex]);
  };

  if (!reportData) {
    return <div>Loading...</div>;
  }

  const mostComplex = reportData.cards.most_complex || {
    category: "N/A",
    subcategory: "N/A",
    u_symptom: "N/A",
    count: 0,
  };

  const dateRange = reportData.cards.date_range || {
    min_date: new Date().toISOString(),
    max_date: new Date().toISOString(),
  };

  const avgResolutionTime = reportData.cards.avg_resolution_time || {
    avg_resolution_time: 0,
  };

  const sortedData = [...reportData.table_data].sort((a, b) => {
    if (!sortField) return 0;
    const aValue = a[sortField];
    const bValue = b[sortField];
    if (aValue < bValue) return sortDirection === "asc" ? -1 : 1;
    if (aValue > bValue) return sortDirection === "asc" ? 1 : -1;
    return 0;
  });

  const paginatedData = sortedData.slice(
    (page - 1) * rowsPerPage,
    page * rowsPerPage
  );

  const handleCloseModal = () => {
    if (!isGeneratingSOP) {
      onClose();
    }
  };

  const overallScore = reportData?.cards?.overall_score || 0;
  const scoreColor = getScoreColor(overallScore);

  const formatNumber = (num: number): string => {
    return new Intl.NumberFormat("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(num);
  };

  return (
    <div className="flex flex-col min-h-screen bg-background">
      <Navbar />
      <main className="flex-grow container mx-auto px-4 py-8">
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-2xl font-bold">Incident Report</h1>
          <Button isIconOnly variant="light" onPress={onInfoModalOpen}>
            <IconInfoCircle size={20} />
          </Button>
        </div>

        <Modal
          isOpen={isInfoModalOpen}
          onClose={onInfoModalClose}
          size="3xl"
          scrollBehavior="inside"
        >
          <ModalContent>
            <ModalHeader className="flex flex-col gap-1">
              Understanding Your Report
            </ModalHeader>
            <ModalBody>
              <div className="space-y-6">
                <section>
                  <h3 className="text-lg font-semibold mb-2">
                    Data Processing Overview
                  </h3>
                  <p className="text-sm text-default-500">
                    This report analyzes your incident data through multiple
                    dimensions to provide actionable insights. The data
                    undergoes several processing steps including sentiment
                    analysis of resolution notes, time-based calculations, and
                    complexity assessments.
                  </p>
                </section>

                <Divider className="my-4" />

                <section>
                  <h3 className="text-lg font-semibold mb-2">
                    Overall Performance Score (0-100)
                  </h3>
                  <p className="text-sm text-default-500 mb-2">
                    This score combines multiple factors to give a comprehensive
                    view of incident management performance:
                  </p>
                  <div className="space-y-2">
                    <div className="bg-default-50 p-3 rounded-lg">
                      <p className="text-sm font-medium">Score Composition:</p>
                      <ul className="list-disc ml-6 mt-1 text-sm text-default-500">
                        <li>
                          70% based on weighted average of individual scores
                          (1-5 scale):
                          <ul className="list-circle ml-6 mt-1">
                            <li>Resolution Time Score (30%)</li>
                            <li>Reassignment Score (20%)</li>
                            <li>Reopen Score (20%)</li>
                            <li>Sentiment Score (30%)</li>
                          </ul>
                        </li>
                        <li>30% based on SLA compliance rate</li>
                      </ul>
                    </div>

                    <div className="bg-default-50 p-3 rounded-lg mt-3">
                      <p className="text-sm font-medium">
                        Individual Score Calculations:
                      </p>
                      <div className="space-y-2 mt-2">
                        <div>
                          <p className="text-sm font-medium text-primary">
                            Resolution Time Score (1-5):
                          </p>
                          <ul className="list-circle ml-6 text-sm text-default-500">
                            <li>
                              Based on percentile ranking of resolution times
                            </li>
                            <li>5: Fastest 20%</li>
                            <li>4: Next 20%</li>
                            <li>3: Middle 20%</li>
                            <li>2: Next 20%</li>
                            <li>1: Slowest 20%</li>
                          </ul>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-primary">
                            Reassignment Score (1-5):
                          </p>
                          <ul className="list-circle ml-6 text-sm text-default-500">
                            <li>5: No reassignments</li>
                            <li>4: 1 reassignment</li>
                            <li>3: 2 reassignments</li>
                            <li>2: 3-4 reassignments</li>
                            <li>1: 5+ reassignments</li>
                          </ul>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-primary">
                            Reopen Score (1-5):
                          </p>
                          <ul className="list-circle ml-6 text-sm text-default-500">
                            <li>5: No reopens</li>
                            <li>4: 1 reopen</li>
                            <li>3: 2 reopens</li>
                            <li>2: 3 reopens</li>
                            <li>1: 4+ reopens</li>
                          </ul>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-primary">
                            Sentiment Score (1-5):
                          </p>
                          <ul className="list-circle ml-6 text-sm text-default-500">
                            <li>5: Highly Positive</li>
                            <li>4: Positive</li>
                            <li>3: Neutral</li>
                            <li>2: Negative</li>
                            <li>1: Highly Negative</li>
                          </ul>
                        </div>
                      </div>
                    </div>

                    <div className="bg-default-50 p-3 rounded-lg mt-3">
                      <p className="text-sm font-medium">Final Score Ranges:</p>
                      <ul className="list-disc ml-6 mt-1">
                        <li className="text-success text-sm">
                          80-100: Outstanding performance
                        </li>
                        <li className="text-warning text-sm">
                          60-79: Average performance
                        </li>
                        <li className="text-danger text-sm">
                          0-59: Needs improvement
                        </li>
                      </ul>
                    </div>
                  </div>
                </section>

                <Divider className="my-4" />

                <section>
                  <h3 className="text-lg font-semibold mb-2">
                    Customer Satisfaction Scoring
                  </h3>
                  <p className="text-sm text-default-500 mb-2">
                    Customer satisfaction is evaluated using multiple components
                    that combine to create a comprehensive satisfaction score:
                  </p>

                  <div className="space-y-4">
                    <div className="bg-default-50 p-3 rounded-lg">
                      <h4 className="font-medium text-primary mb-2">
                        Resolution Time Score (1-5)
                      </h4>
                      <p className="text-sm text-default-500 mb-1">
                        Based on percentile ranking of resolution times:
                      </p>
                      <ul className="list-circle ml-6 text-sm text-default-500">
                        <li>5: Fastest 20% resolution</li>
                        <li>4: Next 20% resolution</li>
                        <li>3: Middle 20% resolution</li>
                        <li>2: Next 20% resolution</li>
                        <li>1: Slowest 20% resolution</li>
                      </ul>
                    </div>

                    <div className="bg-default-50 p-3 rounded-lg">
                      <h4 className="font-medium text-primary mb-2">
                        Reassignment Score (1-5)
                      </h4>
                      <p className="text-sm text-default-500 mb-1">
                        Based on number of times an incident is reassigned:
                      </p>
                      <ul className="list-circle ml-6 text-sm text-default-500">
                        <li>5: No reassignments</li>
                        <li>4: 1 reassignment</li>
                        <li>3: 2 reassignments</li>
                        <li>2: 3-4 reassignments</li>
                        <li>1: 5+ reassignments</li>
                      </ul>
                    </div>

                    <div className="bg-default-50 p-3 rounded-lg">
                      <h4 className="font-medium text-primary mb-2">
                        Reopen Score (1-5)
                      </h4>
                      <p className="text-sm text-default-500 mb-1">
                        Based on number of times an incident is reopened:
                      </p>
                      <ul className="list-circle ml-6 text-sm text-default-500">
                        <li>5: No reopens</li>
                        <li>4: 1 reopen</li>
                        <li>3: 2 reopens</li>
                        <li>2: 3 reopens</li>
                        <li>1: 4+ reopens</li>
                      </ul>
                    </div>

                    <div className="bg-default-50 p-3 rounded-lg">
                      <h4 className="font-medium text-primary mb-2">
                        Sentiment Score (1-5)
                      </h4>
                      <p className="text-sm text-default-500 mb-1">
                        Based on AI analysis of resolution notes:
                      </p>
                      <ul className="list-circle ml-6 text-sm text-default-500">
                        <li>
                          5: Highly Positive - Strong satisfaction indicators
                        </li>
                        <li>4: Positive - Clear satisfaction</li>
                        <li>3: Neutral - No clear sentiment</li>
                        <li>2: Negative - Signs of dissatisfaction</li>
                        <li>
                          1: Highly Negative - Strong dissatisfaction indicators
                        </li>
                      </ul>
                    </div>

                    <div className="bg-default-50 p-3 rounded-lg">
                      <h4 className="font-medium text-primary mb-2">
                        Final Satisfaction Score Calculation
                      </h4>
                      <p className="text-sm text-default-500 mb-1">
                        Weighted average of component scores:
                      </p>
                      <ul className="list-disc ml-6 text-sm text-default-500">
                        <li>Resolution Time Score: 30% weight</li>
                        <li>Reassignment Score: 20% weight</li>
                        <li>Reopen Score: 20% weight</li>
                        <li>Sentiment Score: 30% weight</li>
                      </ul>
                      <div className="mt-3 p-2 bg-default-100 rounded">
                        <p className="text-sm font-medium">
                          Final Score Interpretation:
                        </p>
                        <ul className="list-disc ml-6 mt-1">
                          <li className="text-success text-sm">
                            4.2 - 5.0: Highly Positive
                          </li>
                          <li className="text-success text-sm">
                            3.4 - 4.1: Positive
                          </li>
                          <li className="text-warning text-sm">
                            2.6 - 3.3: Neutral
                          </li>
                          <li className="text-danger text-sm">
                            1.8 - 2.5: Negative
                          </li>
                          <li className="text-danger text-sm">
                            1.0 - 1.7: Highly Negative
                          </li>
                        </ul>
                      </div>
                    </div>
                  </div>

                  <div className="mt-4 p-3 bg-default-100 rounded-lg">
                    <p className="text-sm">
                      <strong>Note:</strong> Customer satisfaction scores are
                      used to:
                    </p>
                    <ul className="list-disc ml-6 mt-1 text-sm text-default-500">
                      <li>Identify trends in customer experience</li>
                      <li>Highlight areas needing improvement</li>
                      <li>Track the effectiveness of process changes</li>
                      <li>Guide training and resource allocation decisions</li>
                    </ul>
                  </div>
                </section>

                <Divider className="my-4" />

                <section>
                  <h3 className="text-lg font-semibold mb-2">
                    Incident Complexity
                  </h3>
                  <p className="text-sm text-default-500 mb-2">
                    Incident complexity is determined by combining normalized
                    resolution times and reassignment counts:
                  </p>
                  <div className="bg-default-50 p-3 rounded-lg">
                    <ul className="list-disc ml-6 text-sm text-default-500">
                      <li>
                        <span className="font-medium">Simple:</span> Quick
                        resolution, minimal reassignments
                      </li>
                      <li>
                        <span className="font-medium">Medium:</span> Average
                        resolution time and reassignments
                      </li>
                      <li>
                        <span className="font-medium">Hard:</span> Above average
                        resolution time or reassignments
                      </li>
                      <li>
                        <span className="font-medium">Complex:</span>{" "}
                        Significantly above average in both metrics
                      </li>
                    </ul>
                  </div>
                </section>

                <Divider className="my-4" />

                <section>
                  <h3 className="text-lg font-semibold mb-2">
                    Actionable Insights Generation
                  </h3>
                  <p className="text-sm text-default-500 mb-4">
                    The system analyzes your incident data across multiple
                    dimensions to generate actionable insights. Here's how each
                    insight type is calculated:
                  </p>

                  <div className="space-y-4">
                    <div>
                      <h4 className="font-medium text-primary">
                        Resolution Time Analysis
                      </h4>
                      <p className="text-sm text-default-500">
                        Identifies incidents taking 50% longer than average to
                        resolve. Highlights the most common category for
                        high-resolution incidents and suggests process
                        improvements based on resolution patterns.
                      </p>
                    </div>

                    <Divider className="my-2" />

                    <div>
                      <h4 className="font-medium text-primary">
                        Incident Routing Analysis
                      </h4>
                      <p className="text-sm text-default-500">
                        Examines incidents with more than 2 reassignments.
                        Calculates reassignment rates by category and identifies
                        potential routing rule improvements or training needs.
                      </p>
                    </div>

                    <Divider className="my-2" />

                    <div>
                      <h4 className="font-medium text-primary">
                        Customer Satisfaction Analysis
                      </h4>
                      <p className="text-sm text-default-500">
                        Groups negative feedback by symptom and category.
                        Calculates satisfaction rates and identifies areas with
                        high concentrations of negative feedback for targeted
                        improvement.
                      </p>
                    </div>

                    <Divider className="my-2" />

                    <div>
                      <h4 className="font-medium text-primary">
                        SLA Compliance Analysis
                      </h4>
                      <p className="text-sm text-default-500">
                        Tracks missed SLAs by category and calculates breach
                        rates. Identifies categories with high SLA miss rates
                        and suggests resource allocation improvements.
                      </p>
                    </div>

                    <Divider className="my-2" />

                    <div>
                      <h4 className="font-medium text-primary">
                        Complexity Distribution
                      </h4>
                      <p className="text-sm text-default-500">
                        Analyzes incident complexity based on resolution times
                        and reassignment patterns. Identifies categories with
                        high concentrations of complex incidents for knowledge
                        base and training improvements.
                      </p>
                    </div>

                    <Divider className="my-2" />

                    <div>
                      <h4 className="font-medium text-primary">
                        Workload Distribution
                      </h4>
                      <p className="text-sm text-default-500">
                        Calculates average incidents per resolver and identifies
                        high workload situations:
                      </p>
                      <ul className="list-disc ml-6 mt-1 text-sm text-default-500">
                        <li>
                          Flags resolvers handling 50% more than average load
                        </li>
                        <li>
                          Identifies their most common incident categories
                        </li>
                        <li>High impact if workload is 2x average or more</li>
                        <li>
                          Suggests workload balancing and training opportunities
                        </li>
                      </ul>
                    </div>
                  </div>

                  <div className="mt-4 p-3 bg-default-100 rounded-lg">
                    <p className="text-sm">
                      <strong>Note:</strong> Impact levels (High/Medium/Low) are
                      assigned based on:
                    </p>
                    <ul className="list-disc ml-6 mt-1 text-sm text-default-500">
                      <li>Deviation from average metrics</li>
                      <li>Volume of affected incidents</li>
                      <li>Potential impact on overall performance score</li>
                    </ul>
                  </div>
                </section>
              </div>
            </ModalBody>
            {/* <ModalFooter>
              <Button color="primary" onPress={onInfoModalClose}>
                Close
              </Button>
            </ModalFooter> */}
          </ModalContent>
        </Modal>

        <Card className="mb-6 bg-background relative overflow-hidden">
          <div className="absolute inset-0">
            {init && (
              <Particles
                id="scoreCardParticles"
                options={particlesOptions}
                className="absolute inset-0"
              />
            )}
          </div>
          <CardBody className="flex flex-col items-center py-8 relative z-10">
            <h2 className="text-xl font-semibold mb-4">
              Overall Performance Score
            </h2>
            <div className={`text-6xl font-bold text-${scoreColor}`}>
              {Number.isInteger(overallScore)
                ? Math.floor(overallScore)
                : overallScore.toFixed(1)}
            </div>
            <div className={`text-${scoreColor} mt-2 font-medium`}>
              {getScoreText(overallScore)}
            </div>
          </CardBody>
        </Card>

        {reportData.actionable_insights &&
          reportData.actionable_insights.length > 0 && (
            <div className="mb-6">
              <div className="flex items-center gap-2 mb-4">
                <IconBulb className="text-primary" size={24} />
                <h2 className="text-xl font-semibold">Actionable Insights</h2>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {reportData.actionable_insights.map(
                  (insight: ActionableInsight, index: number) => (
                    <Card
                      key={index}
                      className={`border-l-4 ${
                        insight.impact === "high"
                          ? "border-l-danger"
                          : insight.impact === "medium"
                          ? "border-l-warning"
                          : "border-l-success"
                      }`}
                    >
                      <CardBody className="p-4">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            {insight.impact === "high" && (
                              <IconAlertTriangle
                                className="text-danger"
                                size={20}
                              />
                            )}
                            <h3 className="font-medium text-lg">
                              {insight.category}
                            </h3>
                          </div>
                          <div className="flex items-center gap-1 text-sm">
                            <span
                              className={`font-semibold ${
                                insight.trend === "up"
                                  ? "text-danger"
                                  : "text-success"
                              }`}
                            >
                              {insight.value}
                            </span>
                            {insight.trend === "up" ? (
                              <IconTrendingUp
                                className="text-danger"
                                size={16}
                              />
                            ) : (
                              <IconTrendingDown
                                className="text-success"
                                size={16}
                              />
                            )}
                          </div>
                        </div>
                        <div className="flex items-start gap-2 mt-2">
                          <IconArrowUpRight
                            className="text-primary mt-1 flex-shrink-0"
                            size={16}
                          />
                          <p className="text-default-500 text-sm">
                            {insight.recommendation}
                          </p>
                        </div>
                      </CardBody>
                    </Card>
                  )
                )}
              </div>
            </div>
          )}

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <Card>
            <CardBody className="flex flex-col">
              <div className="flex items-center mb-2">
                <IconDatabase className="mr-2 text-primary" size={24} />
                <h2 className="text-xl font-semibold">Total Incidents</h2>
              </div>
              <p className="text-3xl font-bold text-primary">
                {new Intl.NumberFormat("en-US").format(
                  reportData.cards.total_incidents
                )}
              </p>
            </CardBody>
          </Card>

          <Card>
            <CardBody className="flex flex-col">
              <div className="flex items-center mb-2">
                <IconCalendar className="mr-2 text-primary" size={24} />
                <h2 className="text-xl font-semibold">Date Range</h2>
              </div>
              <div className="flex items-center gap-2 mt-2">
                <div className="flex items-center">
                  {/* <span className="font-medium mr-2">From:</span> */}
                  <span className="text-default-500">
                    {new Date(dateRange.min_date).toLocaleDateString("en-GB", {
                      day: "2-digit",
                      month: "2-digit",
                      year: "numeric",
                    })}
                  </span>
                </div>
                <IconArrowRight className="mx-2 text-default-400" size={20} />
                <div className="flex items-center">
                  {/* <span className="font-medium mr-2">To:</span> */}
                  <span className="text-default-500">
                    {new Date(dateRange.max_date).toLocaleDateString("en-GB", {
                      day: "2-digit",
                      month: "2-digit",
                      year: "numeric",
                    })}
                  </span>
                </div>
              </div>
            </CardBody>
          </Card>

          <Card
            isPressable
            onPress={handleTimeUnitChange}
            className="transition-transform hover:scale-[1.02]"
          >
            <CardBody className="flex flex-col">
              <div className="flex items-center mb-2">
                <IconClock className="mr-2 text-primary" size={24} />
                <h2 className="text-xl font-semibold">
                  Average Resolution Time
                </h2>
                {/* <IconArrowsExchange2
                  className="ml-2 text-default-400 animate-pulse"
                  size={16}
                /> */}
              </div>
              <p className="text-3xl font-bold text-primary">
                <span className="transition-all duration-300 ease-in-out inline-block transform">
                  {formatNumber(
                    convertTime(avgResolutionTime.avg_resolution_time, timeUnit)
                      .value
                  )}
                </span>
                <span className="text-lg font-normal text-default-500 ml-1 transition-all duration-300 ease-in-out inline-block transform">
                  {timeUnit}
                </span>
              </p>
            </CardBody>
          </Card>
        </div>
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          {reportData?.timing_data && (
            <IncidentTimingChart timingData={reportData.timing_data} />
          )}
          {reportData?.sentiment_distribution && (
            <CustomerSatisfactionChart
              sentimentData={reportData.sentiment_distribution}
            />
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <Card
            isPressable
            onPress={() => setShowLeastComplex(!showLeastComplex)}
            className="transition-transform hover:scale-[1.02]"
          >
            <CardBody className="flex flex-col">
              <div className="flex items-center mb-2">
                <IconBrain className="mr-2 text-primary" size={24} />
                <h2 className="text-xl font-semibold">
                  {showLeastComplex ? "Least" : "Most"} Complex Incident Group
                </h2>
              </div>
              <div className="space-y-2">
                <p>
                  <span className="font-medium">Category:</span>{" "}
                  <span className="text-default-500">
                    {showLeastComplex
                      ? reportData.cards.least_complex.category
                      : reportData.cards.most_complex.category}
                  </span>
                </p>
                <p>
                  <span className="font-medium">Subcategory:</span>{" "}
                  <span className="text-default-500">
                    {showLeastComplex
                      ? reportData.cards.least_complex.subcategory
                      : reportData.cards.most_complex.subcategory}
                  </span>
                </p>
                <p>
                  <span className="font-medium">Symptom:</span>{" "}
                  <span className="text-default-500">
                    {showLeastComplex
                      ? reportData.cards.least_complex.u_symptom
                      : reportData.cards.most_complex.u_symptom}
                  </span>
                </p>
                <p>
                  <span className="font-medium">Count:</span>{" "}
                  <span className="text-primary font-bold">
                    {showLeastComplex
                      ? reportData.cards.least_complex.count
                      : reportData.cards.most_complex.count}
                  </span>
                </p>
              </div>
            </CardBody>
          </Card>

          <Card
            isPressable
            onPress={() => setShowPositiveGroups(!showPositiveGroups)}
            className="transition-transform hover:scale-[1.02]"
          >
            <CardBody className="flex flex-col">
              <div className="flex items-center mb-2">
                <IconMoodHappy className="mr-2 text-primary" size={24} />
                <h2 className="text-xl font-semibold">
                  {showPositiveGroups ? "Most Positive" : "Most Negative"}{" "}
                  Incident Group
                </h2>
              </div>
              <div className="space-y-2">
                <p>
                  <span className="font-medium">Category:</span>{" "}
                  <span className="text-default-500">
                    {showPositiveGroups
                      ? reportData.cards.most_positive.category
                      : reportData.cards.most_negative.category}
                  </span>
                </p>
                <p>
                  <span className="font-medium">Subcategory:</span>{" "}
                  <span className="text-default-500">
                    {showPositiveGroups
                      ? reportData.cards.most_positive.subcategory
                      : reportData.cards.most_negative.subcategory}
                  </span>
                </p>
                <p>
                  <span className="font-medium">Symptom:</span>{" "}
                  <span className="text-default-500">
                    {showPositiveGroups
                      ? reportData.cards.most_positive.u_symptom
                      : reportData.cards.most_negative.u_symptom}
                  </span>
                </p>
                <p>
                  <span className="font-medium">Count:</span>{" "}
                  <span
                    className={`font-bold ${
                      showPositiveGroups ? "text-success" : "text-danger"
                    }`}
                  >
                    {showPositiveGroups
                      ? reportData.cards.most_positive.count
                      : reportData.cards.most_negative.count}
                  </span>
                </p>
              </div>
            </CardBody>
          </Card>
        </div>

        <Table aria-label="Incident data table">
          <TableHeader>
            <TableColumn>Category</TableColumn>
            <TableColumn>Subcategory</TableColumn>
            <TableColumn>Symptom</TableColumn>
            <TableColumn
              onClick={() => handleSort("incident_count")}
              className="cursor-pointer"
            >
              Incident Count{" "}
              {sortField === "incident_count" &&
                (sortDirection === "asc" ? "▲" : "▼")}
            </TableColumn>
            <TableColumn
              onClick={() => handleSort("avg_resolution_time")}
              className="cursor-pointer"
            >
              Avg. Resolution Time (min){" "}
              {sortField === "avg_resolution_time" &&
                (sortDirection === "asc" ? "▲" : "▼")}
            </TableColumn>
            <TableColumn>Actions</TableColumn>
          </TableHeader>
          <TableBody>
            {paginatedData.map((row: any, index: number) => (
              <TableRow key={index}>
                <TableCell>{row.category}</TableCell>
                <TableCell>{row.subcategory}</TableCell>
                <TableCell>{row.u_symptom}</TableCell>
                <TableCell>{row.incident_count}</TableCell>
                <TableCell>{row.avg_resolution_time.toFixed(2)}</TableCell>
                <TableCell>
                  <Button
                    size="sm"
                    onClick={() => handleOpenModal(row)}
                    disabled={isGeneratingSOP}
                  >
                    View Details
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>

        <div className="flex justify-center mt-4">
          <Pagination
            total={Math.ceil(reportData.table_data.length / rowsPerPage)}
            page={page}
            onChange={setPage}
          />
        </div>

        <Modal
          isOpen={isOpen}
          onClose={handleCloseModal}
          size="4xl"
          scrollBehavior="inside"
          isDismissable={!isGeneratingSOP}
          isKeyboardDismissDisabled={isGeneratingSOP}
        >
          <ModalContent>
            <ModalHeader className="flex flex-col gap-1">
              Incident Details
            </ModalHeader>
            <ModalBody>
              {selectedRow && (
                <>
                  <h3 className="text-lg font-semibold">Issue Description</h3>
                  <p>{selectedRow.issue_description}</p>
                  <Divider className="my-4" />
                  <h3 className="text-lg font-semibold">Resolution Notes</h3>
                  <p>{selectedRow.resolution_notes}</p>
                  {sop && (
                    <>
                      <Divider className="my-4" />
                      <h3 className="text-lg font-semibold">Generated SOP</h3>
                      <div className="markdown-body">
                        <ReactMarkdown>{sop}</ReactMarkdown>
                      </div>
                    </>
                  )}
                </>
              )}
            </ModalBody>
            <ModalFooter>
              {!sop ? (
                <Button
                  color="primary"
                  onClick={handleGenerateSOP}
                  isLoading={isGeneratingSOP}
                >
                  {isGeneratingSOP ? "Generating..." : "Generate SOP"}
                </Button>
              ) : (
                <Button
                  color="primary"
                  onClick={handleCopySOP}
                  startContent={
                    isCopied ? <IconCheck size={18} /> : <IconCopy size={18} />
                  }
                >
                  {isCopied ? "Copied!" : "Copy SOP"}
                </Button>
              )}
              {/* {!isGeneratingSOP && (
                <Button color="secondary" onClick={onClose}>
                  Close
                </Button>
              )} */}
            </ModalFooter>
          </ModalContent>
        </Modal>
      </main>
    </div>
  );
};

export default Report;

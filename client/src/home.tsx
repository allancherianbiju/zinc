"use client";

import { useState, useEffect } from "react";
import { useGoogleLogin } from "@react-oauth/google";
import {
  IconBrandGoogleFilled,
  IconUpload,
  IconDatabase,
  IconPlayerPlay,
  IconLogout,
  IconInfoCircle,
  IconSun,
  IconMoon,
  IconCirclePlus,
} from "@tabler/icons-react";
import { useTheme } from "next-themes";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { toast } from "sonner";
import dataImage from "./assets/pexels-pixabay-210607.jpg";
import connectImage from "./assets/pexels-cookiecutter-1148820.jpg";
import demoImage from "./assets/pexels-energepic-com-27411-159888.jpg";

import {
  Button,
  Card,
  CardBody,
  CardHeader,
  CardFooter,
  Image,
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  useDisclosure,
  Checkbox,
  Input,
  Chip,
  Tooltip,
  Link,
  Divider,
  Dropdown,
  DropdownTrigger,
  DropdownMenu,
  DropdownItem,
  Avatar,
  RadioGroup,
  Radio,
  Select,
  SelectItem,
} from "@nextui-org/react";

import api from "./api";

type User = {
  email: string;
  name: string;
  picture: string;
};

type Engagement = {
  id: string;
  name: string;
  userId: string;
};

export default function Component() {
  const { isOpen, onOpen, onOpenChange } = useDisclosure();
  const [isLoading, setIsLoading] = useState(false);
  const [showFullName, setShowFullName] = useState(true);
  const [file, setFile] = useState<File | null>(null);
  const [agreedToTerms, setAgreedToTerms] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const { theme, setTheme } = useTheme();
  const navigate = useNavigate();
  const [engagements, setEngagements] = useState<Engagement[]>([]);
  const [selectedEngagement, setSelectedEngagement] = useState<
    "new" | "existing"
  >("new");
  const [engagementName, setEngagementName] = useState("");
  const [existingEngagementId, setExistingEngagementId] = useState("");

  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  useEffect(() => {
    const timer = setInterval(() => {
      setShowFullName((prev) => !prev);
    }, 5000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const fetchEngagements = async () => {
      if (user) {
        try {
          const response = await api.get("/engagements");
          setEngagements(response.data);
        } catch (error) {
          console.error("Failed to fetch engagements:", error);
        }
      }
    };
    fetchEngagements();
  }, [user]);

  const login = useGoogleLogin({
    onSuccess: async (response) => {
      try {
        const res = await api.post("/auth/google", {
          token: response.access_token,
        });
        const userData = res.data.user;
        setUser(userData);
        localStorage.setItem("user", JSON.stringify(userData));
      } catch (error) {
        console.error("Login error:", error);
      }
    },
    onError: () => console.log("Login Failed"),
  });

  const handleSignOut = () => {
    setUser(null);
    localStorage.removeItem("user");
    api.post("/auth/logout");
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setFile(event.target.files?.[0] || null);
  };

  const handleUpload = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsLoading(true);

    try {
      const formData = new FormData();
      if (file) {
        formData.append("file", file);
      }
      formData.append("engagement_type", selectedEngagement);
      if (selectedEngagement === "new") {
        formData.append("engagement_name", engagementName);
      } else {
        formData.append("engagement_id", existingEngagementId);
      }

      const response = await api.post("/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      toast.success("File uploaded successfully.");
      onOpenChange();
      navigate("/preview", { state: { fileData: response.data } });
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const errorMessage = error.response?.data?.detail || "Upload failed";

        if (errorMessage.includes("already exists")) {
          toast.error(
            "Engagement name already exists. Please choose a different name."
          );
        } else if (errorMessage.includes("Invalid file format")) {
          toast.error("Please upload a valid CSV file.");
        } else if (error.response?.status === 413) {
          toast.error("File size too large. Please upload a smaller file.");
        } else {
          toast.error(errorMessage);
        }

        console.error("Upload error:", error.response?.data);
      } else {
        toast.error("An unexpected error occurred. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-background">
      <main className="flex-grow">
        <section className="w-full py-12 md:py-24 lg:py-32 xl:py-40 bg-gradient-to-b from-secondary-700 to-background dark:from-secondary-100 dark:to-background">
          <div className="container mx-auto px-4">
            <div className="flex flex-col items-center space-y-4 text-center">
              <div className="space-y-2">
                <h1 className="text-6xl font-bold tracking-tighter sm:text-8xl text-primary-foreground relative h-20 sm:h-28">
                  <span
                    className={`absolute inset-0 transition-opacity duration-500 ease-in-out ${
                      showFullName ? "opacity-100" : "opacity-0"
                    }`}
                  >
                    Zinc
                  </span>
                  <span
                    className={`absolute left-1/2 transform -translate-x-1/2 transition-opacity duration-500 ease-in-out ${
                      showFullName ? "opacity-0" : "opacity-100"
                    }  whitespace-nowrap`}
                  >
                    Zero Incidents
                  </span>
                </h1>
                <p className="text-lg text-primary-foreground/80 mt-8">
                  Towards a zero incident future
                </p>
              </div>
            </div>
          </div>
        </section>

        <section
          className={`w-full ${
            user ? "py-16 md:py-20 lg:py-9" : "py-8 md:py-12 lg:py-14"
          }`}
        >
          <div className="container mx-auto px-4">
            <div
              className={`flex flex-col items-center ${
                user ? "mb-16" : "mb-10"
              }`}
            >
              <Divider className="w-1/2 mb-6" />
              {user && user.picture && (
                <div className="mb-10">
                  <Dropdown>
                    <DropdownTrigger>
                      <Avatar
                        isBordered
                        className="transition-transform"
                        name={user.name}
                        src={user.picture}
                      />
                    </DropdownTrigger>
                    <DropdownMenu
                      aria-label="Profile Actions"
                      variant="flat"
                      // disabledKeys={["profile"]}
                    >
                      <DropdownItem key="profile" className="h-14 gap-2">
                        <p className="font-semibold">Signed in as</p>
                        <p className="font-semibold">{user.email}</p>
                      </DropdownItem>
                      <DropdownItem
                        key="theme"
                        onPress={() =>
                          setTheme(theme === "light" ? "dark" : "light")
                        }
                      >
                        <div className="flex items-center gap-2">
                          {theme === "light" ? (
                            <IconMoon size={18} />
                          ) : (
                            <IconSun size={18} />
                          )}
                          Switch to {theme === "light" ? "Dark" : "Light"} mode
                        </div>
                      </DropdownItem>
                      <DropdownItem
                        key="logout"
                        color="danger"
                        onPress={handleSignOut}
                      >
                        <div className="flex items-center gap-2">
                          <IconLogout size={18} />
                          Log Out
                        </div>
                      </DropdownItem>
                    </DropdownMenu>
                  </Dropdown>
                </div>
              )}
              {!user && (
                <>
                  <p className="text-md text-primary-foreground/60 mb-4">
                    Sign in to get started with Zinc
                  </p>
                  <Button
                    color="primary"
                    startContent={<IconBrandGoogleFilled className="w-5 h-5" />}
                    onPress={() => login()}
                  >
                    Sign in with Google
                  </Button>
                </>
              )}
            </div>
            {user && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                <Card isFooterBlurred className="w-full h-[250px]">
                  <CardHeader className="absolute z-10 top-1 flex-col items-start">
                    <p className="text-tiny text-white/60 uppercase font-bold">
                      Upload Dataset
                    </p>
                    <h4 className="text-white/90 font-medium text-xl">
                      Analyze your own data{" "}
                    </h4>
                  </CardHeader>
                  <Image
                    removeWrapper
                    alt="Card example background"
                    className="z-0 w-full h-full scale-125 -translate-y-6 object-cover"
                    src={dataImage}
                  />
                  <CardBody></CardBody>
                  <CardFooter className="absolute bg-white/30 bottom-0 border-t-1 border-zinc-100/50 z-10 justify-between">
                    <div>
                      <p className="text-white/60">
                        Upload your Engagement Data
                      </p>
                    </div>
                    <Button
                      // className="text-tiny"
                      startContent={<IconUpload className="w-4 h-4" />}
                      onPress={onOpen}
                      isDisabled={!user}
                      radius="full"
                      // size="sm"
                    >
                      Upload File
                    </Button>
                    <Modal
                      backdrop="blur"
                      isOpen={isOpen}
                      onOpenChange={onOpenChange}
                      placement="top-center"
                    >
                      <ModalContent>
                        {(onClose) => (
                          <>
                            <ModalHeader className="flex items-center gap-1">
                              Upload Dataset
                              <Tooltip
                                content={
                                  <div className="px-1 py-2">
                                    <div className="text-small font-bold">
                                      Important Information
                                    </div>
                                    <li>Please select a CSV file to upload.</li>
                                    <li>
                                      The selected dataset should only contain
                                      closed incidents.
                                    </li>
                                    <li>
                                      Ensure that the required columns are
                                      present.
                                    </li>
                                  </div>
                                }
                              >
                                <IconInfoCircle className="w-5 h-5" />
                              </Tooltip>
                            </ModalHeader>
                            <form
                              onSubmit={handleUpload}
                              encType="multipart/form-data"
                            >
                              <ModalBody>
                                <RadioGroup
                                  label="Select Engagement"
                                  value={selectedEngagement}
                                  onValueChange={(value) =>
                                    setSelectedEngagement(
                                      value as "new" | "existing"
                                    )
                                  }
                                  orientation="horizontal"
                                >
                                  <Radio value="new">Create New</Radio>
                                  <Radio
                                    value="existing"
                                    isDisabled={engagements.length === 0}
                                    description={
                                      engagements.length === 0
                                        ? "No existing engagements found"
                                        : undefined
                                    }
                                  >
                                    Select Existing
                                  </Radio>
                                </RadioGroup>

                                {selectedEngagement === "new" ? (
                                  <Input
                                    label="Engagement Name"
                                    placeholder="Enter engagement name"
                                    value={engagementName}
                                    onValueChange={setEngagementName}
                                    isRequired
                                    description="This name should be unique and will be used to identify your engagement"
                                  />
                                ) : (
                                  <Select
                                    label="Select Existing Engagement"
                                    placeholder="Choose an engagement"
                                    selectedKeys={
                                      existingEngagementId
                                        ? [existingEngagementId]
                                        : []
                                    }
                                    onChange={(e) =>
                                      setExistingEngagementId(e.target.value)
                                    }
                                    isRequired
                                    isDisabled={engagements.length === 0}
                                    description="Select from one of your existing engagements. The uploaded dataset will be added to this engagement."
                                  >
                                    {engagements.map((engagement) => (
                                      <SelectItem
                                        key={engagement.id}
                                        value={engagement.id}
                                      >
                                        {engagement.name}
                                      </SelectItem>
                                    ))}
                                  </Select>
                                )}
                                <Divider className="my-4" />

                                <Input
                                  type="file"
                                  onChange={handleFileChange}
                                  accept=".csv"
                                  isRequired
                                  startContent={
                                    <IconUpload className="w-4 h-4" />
                                  }
                                />

                                <Checkbox
                                  isSelected={agreedToTerms}
                                  onValueChange={setAgreedToTerms}
                                >
                                  I agree to the{" "}
                                  <Link color="primary" href="#" size="sm">
                                    terms and conditions
                                  </Link>
                                </Checkbox>
                              </ModalBody>
                              <ModalFooter>
                                <Button
                                  color="primary"
                                  type="submit"
                                  isLoading={isLoading}
                                  isDisabled={
                                    !agreedToTerms ||
                                    !file ||
                                    (selectedEngagement === "new" &&
                                      !engagementName) ||
                                    (selectedEngagement === "existing" &&
                                      !existingEngagementId)
                                  }
                                >
                                  {isLoading ? "Uploading" : "Upload"}
                                </Button>
                              </ModalFooter>
                            </form>
                          </>
                        )}
                      </ModalContent>
                    </Modal>
                  </CardFooter>
                </Card>
                <Card isFooterBlurred className="w-full h-[250px]">
                  <CardHeader className="absolute z-10 top-1 flex-col items-start">
                    <p className="text-tiny text-white/60 uppercase font-bold">
                      Connect Database
                    </p>
                    <h4 className="text-white/90 font-medium text-xl">
                      Get real-time insights
                    </h4>
                  </CardHeader>
                  <Image
                    removeWrapper
                    alt="Card example background"
                    className="z-0 w-full h-full scale-125 -translate-y-6 object-cover"
                    src={connectImage}
                  />
                  <CardBody></CardBody>
                  <CardFooter className="absolute bg-white/30 bottom-0 border-t-1 border-zinc-100/50 z-10 justify-between">
                    <div>
                      <p className="text-white/60">Stream data in real-time</p>
                    </div>
                    <Button
                      disabled
                      startContent={<IconDatabase className="w-4 h-4" />}
                    >
                      Coming Soon
                    </Button>
                  </CardFooter>
                </Card>
                <Card className="w-full h-[250px]">
                  <CardHeader className="absolute z-10 top-1 flex-col items-start">
                    <p className="text-tiny text-white/60 uppercase font-bold">
                      View Demo
                    </p>
                    <h4 className="text-white/90 font-medium text-xl">
                      See Zinc in action
                    </h4>
                  </CardHeader>
                  <Image
                    removeWrapper
                    alt="Card example background"
                    className="z-0 w-full h-full scale-125 -translate-y-6 object-cover"
                    src={demoImage}
                  />
                  <CardBody></CardBody>
                  <CardFooter className="absolute bg-white/30 bottom-0 border-t-1 border-zinc-100/50 z-10 justify-end">
                    <Button
                      // color="secondary"
                      startContent={<IconPlayerPlay className="w-4 h-4" />}
                      isDisabled={!user}
                    >
                      Start Demo
                    </Button>
                  </CardFooter>
                </Card>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

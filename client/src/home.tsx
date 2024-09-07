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
} from "@tabler/icons-react";
import { useTheme } from "next-themes";

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
} from "@nextui-org/react";

import api from "./api";
import { useToast } from "@/hooks/use-toast";

type User = {
  email: string;
  name: string;
  picture: string;
};

export default function Component() {
  const { isOpen, onOpen, onOpenChange } = useDisclosure();
  const [isLoading, setIsLoading] = useState(false);
  const [showFullName, setShowFullName] = useState(true);
  const [file, setFile] = useState<File | null>(null);
  const [agreedToTerms, setAgreedToTerms] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const { theme, setTheme } = useTheme();
  const { toast } = useToast();

  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  // Add this useEffect hook
  useEffect(() => {
    const timer = setInterval(() => {
      setShowFullName((prev) => !prev);
    }, 5000);
    return () => clearInterval(timer);
  }, []);

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
    if (!file) return;
    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await api.post("/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      console.log(response.data);
      toast({
        title: "Success",
        description: `File uploaded successfully. ${response.data.rows_processed} rows processed.`,
      });
      onOpenChange();
    } catch (error: any) {
      console.error(error);
      toast({
        title: "Error",
        description:
          error.response?.data?.detail ||
          "An error occurred while uploading the file.",
        variant: "destructive",
      });
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
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card className="w-full">
                <CardHeader>
                  <h4 className="text-lg font-semibold">Upload Dataset</h4>
                </CardHeader>
                <CardBody>
                  <p className="mb-4">Analyze your own data</p>
                  <Button
                    color="primary"
                    startContent={<IconUpload className="w-4 h-4" />}
                    onPress={onOpen}
                    isDisabled={!user}
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
                          <form onSubmit={handleUpload}>
                            <ModalBody>
                              <Input
                                type="file"
                                onChange={handleFileChange}
                                accept=".csv"
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
                                isDisabled={!agreedToTerms || !file}
                              >
                                {isLoading ? "Uploading" : "Upload"}
                              </Button>
                            </ModalFooter>
                          </form>
                        </>
                      )}
                    </ModalContent>
                  </Modal>
                </CardBody>
              </Card>
              <Card className="w-full">
                <CardHeader>
                  <h4 className="text-lg font-semibold">Connect Database</h4>
                </CardHeader>
                <CardBody>
                  <p className="mb-4">Stream data in real-time</p>
                  <Button
                    disabled
                    startContent={<IconDatabase className="w-4 h-4" />}
                  >
                    Coming Soon
                  </Button>
                </CardBody>
              </Card>
              <Card className="w-full">
                <CardHeader>
                  <h4 className="text-lg font-semibold">View Demo</h4>
                </CardHeader>
                <CardBody>
                  <p className="mb-4">See Zinc in action</p>
                  <Button
                    color="secondary"
                    startContent={<IconPlayerPlay className="w-4 h-4" />}
                    isDisabled={!user}
                  >
                    Start Demo
                  </Button>
                </CardBody>
              </Card>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

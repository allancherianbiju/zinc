import React, { useState, useEffect } from "react";
import {
  Navbar,
  NavbarBrand,
  NavbarContent,
  NavbarItem,
  Dropdown,
  DropdownTrigger,
  DropdownMenu,
  DropdownItem,
  Avatar,
} from "@nextui-org/react";
import { useTheme } from "next-themes";
import {
  IconSun,
  IconMoon,
  IconLogout,
  IconSettings,
  IconSparkles,
} from "@tabler/icons-react";
import { Link } from "react-router-dom";

export default function AppNavbar() {
  const { theme, setTheme } = useTheme();
  const user = JSON.parse(localStorage.getItem("user") || "null");
  const [engagement, setEngagement] = useState<string>("");

  useEffect(() => {
    const handleStorageChange = () => {
      const currentEngagement = localStorage.getItem("currentEngagement");
      if (currentEngagement) {
        setEngagement(JSON.parse(currentEngagement).name);
      }
    };

    // Initial check
    handleStorageChange();

    // Listen for changes
    window.addEventListener("storage", handleStorageChange);

    return () => {
      window.removeEventListener("storage", handleStorageChange);
    };
  }, []);

  const handleSignOut = () => {
    localStorage.removeItem("user");
    localStorage.removeItem("currentEngagement");
    window.location.href = "/";
  };

  return (
    <Navbar isBordered maxWidth="full">
      <div className="container mx-auto px-4 flex justify-between items-center w-full">
        <NavbarContent justify="start">
          <NavbarBrand>
            <Link to="/" className="font-bold text-inherit no-underline">
              ZINC {engagement && `/ ${engagement}`}
            </Link>
          </NavbarBrand>
        </NavbarContent>

        <NavbarContent justify="end">
          <div className="flex items-center gap-2">
            <IconSparkles size={20} />
            <p className="text-sm font-semibold">AI Insights</p>
          </div>
          {user && (
            <NavbarItem>
              <Dropdown placement="bottom-end">
                <DropdownTrigger>
                  <Avatar
                    isBordered
                    className="transition-transform"
                    name={user.name}
                    size="sm"
                    src={user.picture}
                  />
                </DropdownTrigger>
                <DropdownMenu aria-label="Profile Actions" variant="flat">
                  <DropdownItem key="profile" className="h-14 gap-2">
                    <p className="font-semibold">Signed in as</p>
                    <p className="font-semibold">{user.email}</p>
                  </DropdownItem>
                  <DropdownItem
                    key="theme"
                    startContent={
                      theme === "light" ? <IconMoon /> : <IconSun />
                    }
                    onPress={() =>
                      setTheme(theme === "light" ? "dark" : "light")
                    }
                  >
                    Switch to {theme === "light" ? "Dark" : "Light"} mode
                  </DropdownItem>
                  <DropdownItem
                    key="settings"
                    startContent={<IconSettings />}
                    onPress={() => (window.location.href = "/settings")}
                  >
                    Settings
                  </DropdownItem>
                  <DropdownItem
                    key="logout"
                    color="danger"
                    startContent={<IconLogout />}
                    onPress={handleSignOut}
                  >
                    Log Out
                  </DropdownItem>
                </DropdownMenu>
              </Dropdown>
            </NavbarItem>
          )}
        </NavbarContent>
      </div>
    </Navbar>
  );
}

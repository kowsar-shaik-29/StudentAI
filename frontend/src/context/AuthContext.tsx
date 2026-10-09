"use client";

import React, { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { UserResponse } from "../types/api";
import { api } from "../lib/api";

interface AuthContextType {
 user: UserResponse | null;
 loading: boolean;
 login: (username: string, pass: string) => Promise<void>;
 register: (username: string, pass: string) => Promise<void>;
 logout: () => Promise<void>;
 refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
 const [user, setUser] = useState<UserResponse | null>(null);
 const [loading, setLoading] = useState<boolean>(true);

 const refreshUser = async () => {
 try {
 const currentUser = await api.getMe();
 setUser(currentUser);
 } catch {
 setUser(null);
 } finally {
 setLoading(false);
 }
 };

 useEffect(() => {
 refreshUser();
 }, []);

 const login = async (username: string, pass: string) => {
 const loggedInUser = await api.login(username, pass);
 setUser(loggedInUser);
 };

 const register = async (username: string, pass: string) => {
 const registeredUser = await api.register(username, pass);
 setUser(registeredUser);
 };

 const logout = async () => {
 try {
 await api.logout();
 } catch {
 // ignore
 } finally {
 setUser(null);
 }
 };

 return (
 <AuthContext.Provider
 value={{
 user,
 loading,
 login,
 register,
 logout,
 refreshUser,
 }}
 >
 {children}
 </AuthContext.Provider>
 );
}

export function useAuth() {
 const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
 return context;
}

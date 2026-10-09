import { createContext, useContext } from "react";

export type UnitSystem = "metric" | "imperial";

export const UnitsContext = createContext<UnitSystem>("metric");

export const useUnits = () => useContext(UnitsContext);

export const km = (v: number, u: UnitSystem) => (u === "imperial" ? v * 0.621371 : v);
export const m = (v: number, u: UnitSystem) => (u === "imperial" ? v * 3.28084 : v);
export const kmh = (v: number, u: UnitSystem) => (u === "imperial" ? v * 0.621371 : v);
export const mps = (v: number, u: UnitSystem) => (u === "imperial" ? v * 2.23694 : v);
export const c = (v: number, u: UnitSystem) => (u === "imperial" ? v * 1.8 + 32 : v);

export const distUnit = (u: UnitSystem) => (u === "imperial" ? "mi" : "km");
export const elevUnit = (u: UnitSystem) => (u === "imperial" ? "ft" : "m");
export const speedUnit = (u: UnitSystem) => (u === "imperial" ? "mph" : "km/h");
export const windUnit = (u: UnitSystem) => (u === "imperial" ? "mph" : "m/s");
export const tempUnit = (u: UnitSystem) => (u === "imperial" ? "°F" : "°C");

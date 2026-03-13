import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ProgressStatus =
  | "not_started"
  | "visited"
  | "trained"
  | "assessed"
  | "passed";

export interface DatasetParams {
  n_stations: number;
}

interface AppState {
  datasetParams: DatasetParams;
  setDatasetParams: (params: Partial<DatasetParams>) => void;

  runIds: Record<string, string>;
  setRunId: (module: string, runId: string) => void;
  getRunId: (module: string) => string | undefined;

  progress: Record<string, ProgressStatus>;
  setProgress: (module: string, status: ProgressStatus) => void;
  getProgress: (module: string) => ProgressStatus;
}

export const useStore = create<AppState>()(
  persist(
    (set, get) => ({
      datasetParams: {
        n_stations: 100,
      },
      setDatasetParams: (params) =>
        set((state) => ({
          datasetParams: { ...state.datasetParams, ...params },
        })),

      runIds: {},
      setRunId: (module, runId) =>
        set((state) => ({
          runIds: { ...state.runIds, [module]: runId },
        })),
      getRunId: (module) => get().runIds[module],

      progress: {},
      setProgress: (module, status) =>
        set((state) => ({
          progress: { ...state.progress, [module]: status },
        })),
      getProgress: (module) => get().progress[module] ?? "not_started",
    }),
    {
      name: "pidem-frontend-store",
      partialize: (state) => ({ progress: state.progress }),
    }
  )
);

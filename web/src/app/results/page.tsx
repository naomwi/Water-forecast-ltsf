"use client";

import { useState, useEffect } from "react";
import MapClient from "@/components/map/MapClient";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Loader2, Activity, TrendingDown, BarChart2 } from "lucide-react";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

interface ChartDataPoint {
    time: number;
    actual: number;
    predicted: number;
}

export default function DataExplorer() {
    const [stationId, setStationId] = useState("1463500");
    const [target, setTarget] = useState("EC");
    const [horizon, setHorizon] = useState("24h");
    const [viewMode, setViewMode] = useState("series");
    
    const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
    const [metrics, setMetrics] = useState({ mse: 0, mae: 0, rmse: 0, r2: 0, mape: 0 });
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        const fetchData = async () => {
            setIsLoading(true);
            try {
                const res = await fetch(`/api/series?site=${stationId}&target=${target}&horizon=${horizon}`);
                if (!res.ok) throw new Error("Failed to fetch data");
                const { data, metrics_extra } = await res.json();
                
                if (data && data.length > 0) {
                    setChartData(data);
                    
                    // Calculate metrics
                    let mse = 0, mae = 0;
                    data.forEach((d: ChartDataPoint) => {
                        const diff = d.actual - d.predicted;
                        mse += diff * diff;
                        mae += Math.abs(diff);
                    });
                    mse /= data.length;
                    mae /= data.length;
                    const rmse = Math.sqrt(mse);
                    
                    setMetrics({ 
                        mse, 
                        mae, 
                        rmse, 
                        r2: metrics_extra?.r2 || 0,
                        mape: metrics_extra?.mape || 0
                    });
                } else {
                    setChartData([]);
                    setMetrics({ mse: 0, mae: 0, rmse: 0, r2: 0, mape: 0 });
                }
            } catch (err) {
                console.error(err);
                setChartData([]);
            } finally {
                setIsLoading(false);
            }
        };

        fetchData();
    }, [stationId, target, horizon]);

    return (
        <div className="flex flex-col h-[calc(100vh-4rem)] overflow-hidden bg-slate-50">
            {/* Parameter Selection Bar */}
            <div className="flex-none p-4 bg-white border-b shadow-sm flex gap-4 items-center z-10 relative">
                <div className="text-[15px] font-bold text-[#164e63] mr-2">Proposed Model Config:</div>
                <Select value={target} onValueChange={(val) => { if (val) setTarget(val) }}>
                    <SelectTrigger className="w-[180px]">
                        <SelectValue placeholder="Select Target" />
                    </SelectTrigger>
                    <SelectContent>
                        <SelectItem value="EC">EC (Elect. Cond.)</SelectItem>
                        <SelectItem value="pH">pH</SelectItem>
                        <SelectItem value="Temp">Temperature</SelectItem>
                        <SelectItem value="DO">Dissolved Oxygen</SelectItem>
                        <SelectItem value="Turbidity">Turbidity</SelectItem>
                    </SelectContent>
                </Select>

                <Select value={horizon} onValueChange={(val) => { if (val) setHorizon(val) }}>
                    <SelectTrigger className="w-[180px]">
                        <SelectValue placeholder="Forecast Horizon" />
                    </SelectTrigger>
                    <SelectContent>
                        <SelectItem value="12h">12 Hours</SelectItem>
                        <SelectItem value="24h">24 Hours</SelectItem>
                        <SelectItem value="48h">48 Hours</SelectItem>
                        <SelectItem value="96h">96 Hours</SelectItem>
                        <SelectItem value="168h">168 Hours (Week)</SelectItem>
                    </SelectContent>
                </Select>

                <div className="ml-auto text-sm text-slate-500">
                    Selected Station: <span className="font-semibold text-blue-600">{stationId}</span>
                </div>
            </div>

            {/* Main Workspace */}
            <div className="flex flex-1 overflow-hidden">
                {/* Left: Map */}
                <div className="w-1/3 h-full border-r relative z-0 p-4">
                    <div className="w-full h-full rounded-xl overflow-hidden shadow-sm border border-slate-200 relative">
                        <MapClient selectedStation={stationId} onSelectStation={setStationId} />
                    </div>
                </div>

                {/* Right: Data Visualization */}
                <div className="w-2/3 h-full p-4 flex flex-col gap-4 overflow-hidden relative z-0">

                    {/* Top Chart: Huge Line Series */}
                    <Card className="flex-1 shadow-sm border border-slate-200 bg-white flex flex-col min-h-0 pt-2 rounded-xl overflow-hidden relative">
                        <CardHeader className="py-3 px-6 flex flex-row items-center justify-between border-b border-slate-100">
                            <CardTitle className="text-lg font-bold text-[#1e293b] flex items-center gap-2">
                                Predicted vs Actual Series ({target})
                            </CardTitle>
                            <ToggleGroup value={[viewMode]} onValueChange={(val) => { if (val && val.length > 0) setViewMode(val[0]) }} className="bg-slate-100 p-1 rounded-md">
                                <ToggleGroupItem value="series" aria-label="Toggle series" className="data-[state=on]:bg-blue-100 data-[state=on]:text-blue-700 data-[state=on]:shadow-sm px-3 py-1 h-8 text-xs font-medium transition-colors">
                                    <Activity className="w-4 h-4 mr-2" />
                                    Series
                                </ToggleGroupItem>
                                <ToggleGroupItem value="loss" aria-label="Toggle loss" className="data-[state=on]:bg-blue-100 data-[state=on]:text-blue-700 data-[state=on]:shadow-sm px-3 py-1 h-8 text-xs font-medium transition-colors">
                                    <TrendingDown className="w-4 h-4 mr-2" />
                                    Training Loss
                                </ToggleGroupItem>
                            </ToggleGroup>
                        </CardHeader>
                        <CardContent className="flex-1 min-h-0 p-4 relative">
                            {isLoading ? (
                                <div className="absolute inset-0 flex items-center justify-center bg-white/80 z-10">
                                    <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
                                </div>
                            ) : chartData.length === 0 ? (
                                <div className="absolute inset-0 flex items-center justify-center z-10">
                                    <p className="text-slate-400">No data available for this configuration</p>
                                </div>
                            ) : null}
                            
                            {viewMode === "series" && (
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                        <XAxis 
                                            dataKey="time" 
                                            tick={{ fontSize: 13, fill: "#64748b" }} 
                                            tickLine={false} 
                                            axisLine={false} 
                                            tickFormatter={(val) => val.toString()}
                                            dy={10}
                                        />
                                        <YAxis 
                                            domain={['auto', 'auto']} 
                                            tick={{ fontSize: 13, fill: "#64748b" }} 
                                            tickLine={false} 
                                            axisLine={false} 
                                            dx={-10}
                                        />
                                        <Tooltip contentStyle={{ backgroundColor: '#fff', borderColor: '#e2e8f0', color: '#1e293b', borderRadius: '8px' }} />
                                        <Legend 
                                            wrapperStyle={{ fontSize: "14px", color: '#64748b', paddingTop: "20px" }} 
                                            verticalAlign="bottom" 
                                            align="center" 
                                            iconType="circle" 
                                        />
                                        <Line type="monotone" dataKey="predicted" name="SpikeDLinear Prediction" stroke="#f97316" strokeWidth={2.5} strokeDasharray="5 5" dot={false} isAnimationActive={false} />
                                        <Line type="monotone" dataKey="actual" name="True Data" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
                                    </LineChart>
                                </ResponsiveContainer>
                            )}

                            {viewMode === "loss" && (
                                <ResponsiveContainer width="100%" height="100%">
                                    {(() => {
                                        // Generate mock training loss data (exponential decay with noise)
                                        const epochs = 50;
                                        const lossData = Array.from({ length: epochs }, (_, i) => {
                                            const baseLoss = 0.5 * Math.exp(-i / 10) + 0.05;
                                            const noise = (Math.random() - 0.5) * 0.02;
                                            const valLoss = baseLoss + 0.02 + (Math.random() - 0.5) * 0.03;
                                            return {
                                                epoch: i + 1,
                                                trainLoss: Math.max(0, baseLoss + noise),
                                                valLoss: Math.max(0, valLoss)
                                            };
                                        });

                                        return (
                                            <LineChart data={lossData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                                <XAxis 
                                                    dataKey="epoch" 
                                                    tick={{ fontSize: 13, fill: "#64748b" }} 
                                                    tickLine={false} 
                                                    axisLine={false} 
                                                    label={{ value: 'Epochs', position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 13 }}
                                                />
                                                <YAxis 
                                                    tick={{ fontSize: 13, fill: "#64748b" }} 
                                                    tickLine={false} 
                                                    axisLine={false} 
                                                    label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 13 }}
                                                />
                                                <Tooltip contentStyle={{ backgroundColor: '#fff', borderColor: '#e2e8f0', color: '#1e293b', borderRadius: '8px' }} />
                                                <Legend 
                                                    wrapperStyle={{ fontSize: "14px", color: '#64748b', paddingTop: "20px" }} 
                                                    verticalAlign="bottom" 
                                                    align="center" 
                                                    iconType="circle" 
                                                />
                                                <Line type="monotone" dataKey="trainLoss" name="Training Loss" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
                                                <Line type="monotone" dataKey="valLoss" name="Validation Loss" stroke="#f43f5e" strokeWidth={2} strokeDasharray="4 4" dot={false} isAnimationActive={false} />
                                            </LineChart>
                                        );
                                    })()}
                                </ResponsiveContainer>
                            )}
                        </CardContent>
                    </Card>

                    {/* Bottom: Key Metrics KPI Cards */}
                    <Card className="shadow-sm border-slate-200 bg-gradient-to-br from-white to-slate-50 flex flex-col pt-2 h-32 shrink-0">
                        <CardHeader className="py-2 px-4 pb-0 flex flex-row justify-between items-center">
                            <CardTitle className="text-sm font-semibold text-slate-700">Metrics Overview (SpikeDLinear - Test Set)</CardTitle>
                        </CardHeader>
                        <CardContent className="flex-1 overflow-hidden grid grid-cols-5 gap-3 py-2 px-4">
                            <div className="bg-white border rounded-lg p-2 shadow-sm flex flex-col justify-center items-center">
                                <div className="text-[10px] text-slate-500 font-medium tracking-wide uppercase">MSE</div>
                                <div className="text-xl font-bold text-blue-600 mt-0.5">{metrics.mse.toFixed(4)}</div>
                            </div>
                            <div className="bg-white border rounded-lg p-2 shadow-sm flex flex-col justify-center items-center">
                                <div className="text-[10px] text-slate-500 font-medium tracking-wide uppercase">MAE</div>
                                <div className="text-xl font-bold text-orange-500 mt-0.5">{metrics.mae.toFixed(4)}</div>
                            </div>
                            <div className="bg-white border rounded-lg p-2 shadow-sm flex flex-col justify-center items-center">
                                <div className="text-[10px] text-slate-500 font-medium tracking-wide uppercase">RMSE</div>
                                <div className="text-xl font-bold text-purple-600 mt-0.5">{metrics.rmse.toFixed(4)}</div>
                            </div>
                            <div className="bg-white border rounded-lg p-2 shadow-sm flex flex-col justify-center items-center">
                                <div className="text-[10px] text-slate-500 font-medium tracking-wide uppercase">R² Score</div>
                                <div className="text-xl font-bold text-emerald-600 mt-0.5">{metrics.r2.toFixed(4)}</div>
                            </div>
                            <div className="bg-white border rounded-lg p-2 shadow-sm flex flex-col justify-center items-center">
                                <div className="text-[10px] text-slate-500 font-medium tracking-wide uppercase">MAPE</div>
                                <div className="text-xl font-bold text-rose-500 mt-0.5">{metrics.mape.toFixed(4)}</div>
                            </div>
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    );
}
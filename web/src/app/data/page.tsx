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
}

export default function RawDataExplorer() {
    const [stationId, setStationId] = useState("1463500");
    const [target, setTarget] = useState("EC");
    const [viewMode, setViewMode] = useState("series");
    
    const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
    const [stats, setStats] = useState({ mean: 0, min: 0, max: 0, std: 0, count: 0 });
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        const fetchData = async () => {
            setIsLoading(true);
            try {
                const res = await fetch(`/api/raw?site=${stationId}&target=${target}`);
                if (!res.ok) throw new Error("Failed to fetch data");
                const { data, stats: newStats } = await res.json();
                
                if (data && data.length > 0) {
                    setChartData(data);
                    if (newStats) {
                        setStats(newStats);
                    }
                } else {
                    setChartData([]);
                    setStats({ mean: 0, min: 0, max: 0, std: 0, count: 0 });
                }
            } catch (err) {
                console.error(err);
                setChartData([]);
            } finally {
                setIsLoading(false);
            }
        };

        fetchData();
    }, [stationId, target]);

    return (
        <div className="flex flex-col h-[calc(100vh-4rem)] overflow-hidden bg-slate-50">
            {/* Parameter Selection Bar */}
            <div className="flex-none p-4 bg-white border-b shadow-sm flex gap-4 items-center z-10 relative">
                <div className="text-[15px] font-bold text-[#164e63] mr-2">Data Configuration:</div>
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

                    {/* Top Chart: Raw Series / Heatmap / Histogram */}
                    <Card className="flex-1 shadow-sm border border-slate-200 bg-white flex flex-col min-h-0 pt-2 rounded-xl overflow-hidden relative">
                        <CardHeader className="py-3 px-6 flex flex-row items-center justify-between border-b border-slate-100">
                            <CardTitle className="text-lg font-bold text-[#1e293b] flex items-center gap-2">
                                Raw Data ({target})
                            </CardTitle>
                            <ToggleGroup value={[viewMode]} onValueChange={(val) => { if (val && val.length > 0) setViewMode(val[0]) }} className="bg-slate-100 p-1 rounded-md">
                                <ToggleGroupItem value="series" aria-label="Toggle series" className="data-[state=on]:bg-blue-100 data-[state=on]:text-blue-700 data-[state=on]:shadow-sm px-3 py-1 h-8 text-xs font-medium transition-colors">
                                    <Activity className="w-4 h-4 mr-2" />
                                    Raw Series
                                </ToggleGroupItem>
                                <ToggleGroupItem value="heatmap" aria-label="Toggle heatmap" className="data-[state=on]:bg-blue-100 data-[state=on]:text-blue-700 data-[state=on]:shadow-sm px-3 py-1 h-8 text-xs font-medium transition-colors">
                                    <TrendingDown className="w-4 h-4 mr-2" />
                                    Heatmap
                                </ToggleGroupItem>
                                <ToggleGroupItem value="histogram" aria-label="Toggle histogram" className="data-[state=on]:bg-blue-100 data-[state=on]:text-blue-700 data-[state=on]:shadow-sm px-3 py-1 h-8 text-xs font-medium transition-colors">
                                    <BarChart2 className="w-4 h-4 mr-2" />
                                    Histogram
                                </ToggleGroupItem>
                            </ToggleGroup>
                        </CardHeader>
                        <CardContent className="flex-1 min-h-0 p-4 relative">
                            {isLoading && viewMode !== "heatmap" ? (
                                <div className="absolute inset-0 flex items-center justify-center bg-white/80 z-10">
                                    <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
                                </div>
                            ) : chartData.length === 0 && viewMode !== "heatmap" ? (
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
                                        <Line type="monotone" dataKey="actual" name={`Historical ${target} (Sample)`} stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
                                    </LineChart>
                                </ResponsiveContainer>
                            )}

                            {viewMode === "heatmap" && (
                                <div className="absolute inset-0 w-full h-full flex flex-col items-center justify-center bg-white p-6 overflow-auto">
                                    <h3 className="text-md font-bold mb-6 text-slate-700">Feature Correlation Matrix</h3>
                                    {(() => {
                                        const features = ["EC", "pH", "Temp", "DO", "Turbidity"];
                                        // Mock correlation matrix for water quality data
                                        const matrix = [
                                            [1.00,  0.15,  0.20, -0.40,  0.65],
                                            [0.15,  1.00,  0.10, -0.10,  0.25],
                                            [0.20,  0.10,  1.00, -0.85,  0.30],
                                            [-0.40, -0.10, -0.85,  1.00, -0.55],
                                            [0.65,  0.25,  0.30, -0.55,  1.00]
                                        ];

                                        const getColor = (val: number) => {
                                            if (val > 0) {
                                                return `rgba(59, 130, 246, ${val})`;
                                            } else {
                                                return `rgba(239, 68, 68, ${Math.abs(val)})`;
                                            }
                                        };

                                        return (
                                            <div className="flex flex-col gap-1">
                                                <div className="flex gap-1">
                                                    <div className="w-16 h-10 flex items-center justify-end pr-2 text-xs font-semibold text-slate-500"></div>
                                                    {features.map(f => (
                                                        <div key={f} className="w-16 h-10 flex items-center justify-center text-xs font-semibold text-slate-500">{f}</div>
                                                    ))}
                                                </div>
                                                {features.map((fRow, i) => (
                                                    <div key={fRow} className="flex gap-1">
                                                        <div className="w-16 h-16 flex items-center justify-end pr-2 text-xs font-semibold text-slate-500">{fRow}</div>
                                                        {features.map((fCol, j) => {
                                                            const val = matrix[i][j];
                                                            const textColor = Math.abs(val) > 0.5 ? "text-white" : "text-slate-800";
                                                            return (
                                                                <div 
                                                                    key={`${fRow}-${fCol}`} 
                                                                    className={`w-16 h-16 flex items-center justify-center text-xs font-medium rounded-sm shadow-sm transition-transform hover:scale-105 cursor-default ${textColor}`}
                                                                    style={{ backgroundColor: getColor(val) }}
                                                                    title={`${fRow} vs ${fCol}: ${val.toFixed(2)}`}
                                                                >
                                                                    {val.toFixed(2)}
                                                                </div>
                                                            )
                                                        })}
                                                    </div>
                                                ))}
                                                <div className="mt-8 flex items-center gap-2 justify-center text-xs text-slate-500">
                                                    <span>-1.0</span>
                                                    <div className="w-32 h-3 bg-gradient-to-r from-red-500 via-white to-blue-500 rounded-sm border border-slate-200"></div>
                                                    <span>1.0</span>
                                                </div>
                                            </div>
                                        );
                                    })()}
                                </div>
                            )}

                            {viewMode === "histogram" && (
                                <ResponsiveContainer width="100%" height="100%">
                                    {(() => {
                                        if (chartData.length === 0) return <div />;
                                        // Calculate distribution of actual values
                                        const values = chartData.map(d => d.actual);
                                        const minVal = Math.min(...values);
                                        const maxVal = Math.max(...values);
                                        const binCount = 30;
                                        // Handle edge case where max == min
                                        const binSize = maxVal > minVal ? (maxVal - minVal) / binCount : 1;
                                        
                                        const bins = Array.from({ length: binCount }, (_, i) => ({
                                            binStart: minVal + i * binSize,
                                            binEnd: minVal + (i + 1) * binSize,
                                            count: 0
                                        }));

                                        values.forEach(val => {
                                            let binIdx = Math.floor((val - minVal) / binSize);
                                            if (binIdx >= binCount) binIdx = binCount - 1;
                                            bins[binIdx].count++;
                                        });

                                        const histogramData = bins.map(b => ({
                                            range: `${b.binStart.toFixed(1)} to ${b.binEnd.toFixed(1)}`,
                                            midpoint: ((b.binStart + b.binEnd) / 2).toFixed(1),
                                            count: b.count
                                        }));

                                        return (
                                            <BarChart data={histogramData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                                <XAxis 
                                                    dataKey="midpoint" 
                                                    tick={{ fontSize: 12, fill: "#64748b" }} 
                                                    tickLine={false} 
                                                    axisLine={false}
                                                    label={{ value: `${target} Values`, position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 13 }}
                                                />
                                                <YAxis 
                                                    tick={{ fontSize: 12, fill: "#64748b" }} 
                                                    tickLine={false} 
                                                    axisLine={false}
                                                    label={{ value: 'Frequency', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 13 }}
                                                />
                                                <Tooltip 
                                                    contentStyle={{ backgroundColor: '#fff', borderColor: '#e2e8f0', color: '#1e293b', borderRadius: '8px' }}
                                                    cursor={{ fill: '#f1f5f9' }}
                                                />
                                                <Bar dataKey="count" name="Frequency" fill="#3b82f6" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                                            </BarChart>
                                        );
                                    })()}
                                </ResponsiveContainer>
                            )}
                        </CardContent>
                    </Card>

                    {/* Bottom: Data Overview Cards */}
                    <Card className="shadow-sm border-slate-200 bg-gradient-to-br from-white to-slate-50 flex flex-col pt-2 h-32 shrink-0">
                        <CardHeader className="py-2 px-4 pb-0 flex flex-row justify-between items-center">
                            <CardTitle className="text-sm font-semibold text-slate-700">Data Overview ({target})</CardTitle>
                        </CardHeader>
                        <CardContent className="flex-1 overflow-hidden grid grid-cols-5 gap-3 py-2 px-4">
                            <div className="bg-white border rounded-lg p-2 shadow-sm flex flex-col justify-center items-center">
                                <div className="text-[10px] text-slate-500 font-medium tracking-wide uppercase">Mean</div>
                                <div className="text-xl font-bold text-slate-800 mt-0.5">{stats.mean.toFixed(2)}</div>
                            </div>
                            <div className="bg-white border rounded-lg p-2 shadow-sm flex flex-col justify-center items-center">
                                <div className="text-[10px] text-slate-500 font-medium tracking-wide uppercase">Min</div>
                                <div className="text-xl font-bold text-slate-800 mt-0.5">{stats.min.toFixed(2)}</div>
                            </div>
                            <div className="bg-white border rounded-lg p-2 shadow-sm flex flex-col justify-center items-center">
                                <div className="text-[10px] text-slate-500 font-medium tracking-wide uppercase">Max</div>
                                <div className="text-xl font-bold text-slate-800 mt-0.5">{stats.max.toFixed(2)}</div>
                            </div>
                            <div className="bg-white border rounded-lg p-2 shadow-sm flex flex-col justify-center items-center">
                                <div className="text-[10px] text-slate-500 font-medium tracking-wide uppercase">Std Dev</div>
                                <div className="text-xl font-bold text-slate-800 mt-0.5">{stats.std.toFixed(2)}</div>
                            </div>
                            <div className="bg-white border rounded-lg p-2 shadow-sm flex flex-col justify-center items-center">
                                <div className="text-[10px] text-slate-500 font-medium tracking-wide uppercase">Count</div>
                                <div className="text-xl font-bold text-slate-800 mt-0.5">{stats.count.toLocaleString()}</div>
                            </div>
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    );
}

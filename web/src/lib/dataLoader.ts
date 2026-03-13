import fs from 'fs';
import path from 'path';
import zlib from 'zlib';
import Papa from 'papaparse';

// ============================================================================
// TYPES & INTERFACES (Clean Architecture / Lint & Validate)
// ============================================================================

interface RawDataRow {
    site_no: number;
    Time?: string;
    [key: string]: unknown;
}

interface MetricsRow {
    R2?: number;
    MAPE?: number;
    [key: string]: unknown;
}

interface SeriesRow {
    Actual: number;
    Predicted: number;
    Time?: string;
    [key: string]: unknown;
}

const PROJECT_DIR = path.resolve(process.cwd(), 'data');

// ============================================================================
// CORE DATA LOADERS (No Python needed)
// ============================================================================

export async function getRawData(site: string, target: string) {
    const csvPath = path.join(PROJECT_DIR, 'USGs', 'water_data_sample.csv.gz');
    
    if (!fs.existsSync(csvPath)) {
        return { success: false, error: `Dataset not found at ${csvPath}` };
    }

    try {
        const compressed = fs.readFileSync(csvPath);
        const uncompressed = zlib.gunzipSync(compressed).toString('utf-8');
        
        const parseResult = Papa.parse<RawDataRow>(uncompressed, {
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true,
        });

        // Filter by site
        const siteNo = parseInt(site, 10);
        const df = parseResult.data.filter(row => row.site_no === siteNo);

        if (df.length === 0) {
            return { success: false, error: `No data found for site ${site}` };
        }

        // Sort by Time
        if (df.length > 0 && df[0].Time) {
            df.sort((a, b) => (a.Time || '').localeCompare(b.Time || ''));
        }

        // Extract valid numbers
        const targetSeries = df.map(row => row[target]).filter(val => typeof val === 'number' && !isNaN(val)) as number[];
        
        const count = targetSeries.length;
        const mean = count > 0 ? targetSeries.reduce((a, b) => a + b, 0) / count : 0;
        const min = count > 0 ? Math.min(...targetSeries) : 0;
        const max = count > 0 ? Math.max(...targetSeries) : 0;
        
        let variance = 0;
        if (count > 1) {
            variance = targetSeries.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (count - 1);
        }
        const std = Math.sqrt(variance);

        const stats = { mean, min, max, std, count };

        // Downsample every 15th row to keep UI fast
        const data = [];
        for (let i = 0; i < df.length; i += 15) {
            const row = df[i];
            const val = row[target];
            if (typeof val === 'number' && !isNaN(val)) {
                const timeStr = row.Time || String(i);
                const displayTime = timeStr.length >= 7 ? timeStr.substring(0, 7) : String(i);
                data.push({
                    time: displayTime,
                    actual: val
                });
            }
        }

        return { success: true, data, stats };

    } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : 'Unknown error';
        return { success: false, error: msg };
    }
}

export async function getSeriesData(site: string, target: string, horizon: string) {
    try {
        const metricsPath = path.join(PROJECT_DIR, 'Proposed_Models', target, 'results', `site_${site}`, 'metrics', `SpikeDLinear_h${horizon}.csv`);
        
        let r2 = 0;
        let mape = 0;

        if (fs.existsSync(metricsPath)) {
            const metricsContent = fs.readFileSync(metricsPath, 'utf-8');
            const metricsParse = Papa.parse<MetricsRow>(metricsContent, { header: true, skipEmptyLines: true, dynamicTyping: true });
            if (metricsParse.data.length > 0) {
                r2 = metricsParse.data[0].R2 || 0;
                mape = metricsParse.data[0].MAPE || 0;
            }
        }

        const seriesPath = path.join(PROJECT_DIR, 'Proposed_Models', target, 'results', `site_${site}`, 'series', `series_SpikeDLinear_P${horizon}_${target}.csv`);
        
        if (!fs.existsSync(seriesPath)) {
            return { success: false, error: `Series file not found: ${seriesPath}` };
        }

        const seriesContent = fs.readFileSync(seriesPath, 'utf-8');
        const seriesParse = Papa.parse<SeriesRow>(seriesContent, { header: true, skipEmptyLines: true, dynamicTyping: true });
        const seriesDf = seriesParse.data;

        // Load raw data to get timestamps
        const rawPath = path.join(PROJECT_DIR, 'Deep_Baselines', 'data', 'USGs', 'water_data_sample.csv.gz');
        let timestamps: string[] = [];
        
        if (fs.existsSync(rawPath)) {
            const compressed = fs.readFileSync(rawPath);
            const uncompressed = zlib.gunzipSync(compressed).toString('utf-8');
            const rawParse = Papa.parse<RawDataRow>(uncompressed, { header: true, skipEmptyLines: true, dynamicTyping: true });
            
            const rawDf = rawParse.data.filter(r => r.site_no === parseInt(site, 10));
            if (rawDf.length > 0 && rawDf[0].Time) {
                rawDf.sort((a, b) => (a.Time || '').localeCompare(b.Time || ''));
            }
            
            if (rawDf.length >= seriesDf.length) {
                timestamps = rawDf.slice(-seriesDf.length).map(r => r.Time || '');
            }
        }

        // Downsample the series_df for visualization (e.g., max 500 points) to avoid lagging
        const step = Math.max(1, Math.floor(seriesDf.length / 800));
        const data = [];
        
        for (let i = 0; i < seriesDf.length; i += step) {
            const row = seriesDf[i];
            const timeStr = (timestamps[i] || String(i)).substring(0, 10);
            
            data.push({
                time: timeStr,
                actual: Number(row.Actual) || 0,
                predicted: Number(row.Predicted) || 0
            });
        }

        return {
            success: true,
            data,
            metrics_extra: { r2, mape }
        };

    } catch (e: unknown) {
         const msg = e instanceof Error ? e.message : 'Unknown error';
         return { success: false, error: msg };
    }
}
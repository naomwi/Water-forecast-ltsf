import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';

// ============================================================================
// TYPES & INTERFACES (Clean Architecture / Lint & Validate)
// ============================================================================

export interface Intent {
    is_prediction: boolean;
    features: string[];
    horizon: number | null;
    site: number | null;
    raw_query: string;
}

export interface PredictionResult {
    feature: string;
    horizon: number;
    site: number;
    predicted_mean: number;
    predicted_min: number;
    predicted_max: number;
    predicted_last: number;
    actual_last: number;
    n_points: number;
    predicted_values: number[];
}

interface CSVRow {
    Timestep?: string | number;
    Actual: string | number;
    Predicted: string | number;
    [key: string]: unknown;
}

// ============================================================================
// CONSTANTS & MAPPINGS
// ============================================================================

const FEATURE_KEYWORDS: Record<string, string[]> = {
    'EC': ['ec', 'electrical conductivity', 'conductivity', 'điện dẫn', 'độ dẫn điện', 'salinity', 'mặn'],
    'pH': ['ph', 'độ ph', 'acid', 'axit'],
    'Temp': ['temp', 'temperature', 'nhiệt độ', 'nhiet do'],
    'Flow': ['flow', 'lưu lượng', 'luu luong', 'dòng chảy', 'dong chay'],
    'DO': ['do', 'dissolved oxygen', 'oxy hòa tan', 'oxy'],
    'Turbidity': ['turbidity', 'độ đục', 'do duc', 'đục'],
};

const WATER_QUALITY_KEYWORDS = [
    'chất lượng nước', 'chat luong nuoc', 'water quality',
    'nước sạch', 'nuoc sach', 'clean water', 'ô nhiễm', 'o nhiem',
    'pollution', 'contamination', 'tổng hợp', 'tong hop', 'overall',
    'all features', 'tất cả', 'tat ca', 'toàn bộ', 'toan bo',
    'predict all', 'dự đoán nước', 'du doan nuoc',
];

const HORIZON_PATTERNS: Record<number, RegExp[]> = {
    6: [/6\s*(?:h|giờ|gio|hour)/i, /6\s*tiếng/i],
    12: [/12\s*(?:h|giờ|gio|hour)/i, /12\s*tiếng/i, /nửa ngày/i, /nua ngay/i],
    24: [/24\s*(?:h|giờ|gio|hour)/i, /1\s*(?:ngày|ngay|day)/i, /một ngày/i, /mot ngay/i],
    48: [/48\s*(?:h|giờ|gio|hour)/i, /2\s*(?:ngày|ngay|day)/i, /hai ngày/i, /hai ngay/i],
    96: [/96\s*(?:h|giờ|gio|hour)/i, /4\s*(?:ngày|ngay|day)/i, /bốn ngày/i, /bon ngay/i],
    168: [/168\s*(?:h|giờ|gio|hour)/i, /7\s*(?:ngày|ngay|day)/i, /(?:1|một|mot)\s*tuần/i, /(?:1|một|mot)\s*tuan/i, /one week/i, /a week/i],
};

const DAY_PATTERN = /(\d+)\s*(?:ngày|ngay|day|days)/i;
const HOUR_PATTERN = /(\d+)\s*(?:giờ|gio|h(?:our)?s?|tiếng|tieng)/i;
const SITE_PATTERN = /(?:trạm|tram|site)\s*(\d+)/i;

const VALID_HORIZONS = [6, 12, 24, 48, 96, 168];
const DEFAULT_SITE = 1463500;

// Project root is one level up from `web`
const PROJECT_DIR = path.resolve(process.cwd(), 'data');

// ============================================================================
// HELPER FUNCTIONS (Debugging Strategies / DRY)
// ============================================================================

function nearestHorizon(hours: number): number {
    return VALID_HORIZONS.reduce((prev, curr) => 
        Math.abs(curr - hours) < Math.abs(prev - hours) ? curr : prev
    );
}

function detectFeatures(messageLower: string): string[] {
    const isWaterQuality = WATER_QUALITY_KEYWORDS.some(kw => messageLower.includes(kw));
    
    if (isWaterQuality) {
        return Object.keys(FEATURE_KEYWORDS);
    }

    const detectedFeatures: string[] = [];
    for (const [feature, keywords] of Object.entries(FEATURE_KEYWORDS)) {
        const hasMatch = keywords.some(kw => {
            // Use word boundary for short keywords to prevent false positives (e.g. "predict" contains "ec")
            if (kw.length <= 2) {
                const regex = new RegExp(`\\b${kw}\\b`, 'i');
                return regex.test(messageLower);
            }
            return messageLower.includes(kw);
        });
        
        if (hasMatch) {
            detectedFeatures.push(feature);
        }
    }
    return detectedFeatures;
}

function detectHorizonValue(messageLower: string): number | null {
    // 1. Check explicit mappings
    for (const [horizonStr, patterns] of Object.entries(HORIZON_PATTERNS)) {
        const horizonVal = parseInt(horizonStr, 10);
        if (patterns.some(pattern => pattern.test(messageLower))) {
            return horizonVal;
        }
    }

    // 2. Check general day/hour patterns (Debugging strategies applied: fallback grouping)
    const dayMatch = messageLower.match(DAY_PATTERN);
    if (dayMatch && dayMatch[1]) {
        const days = parseInt(dayMatch[1], 10);
        return nearestHorizon(days * 24);
    }

    const hourMatch = messageLower.match(HOUR_PATTERN);
    if (hourMatch && hourMatch[1]) {
        const hours = parseInt(hourMatch[1], 10);
        return nearestHorizon(hours);
    }

    return null;
}

// ============================================================================
// CORE EXPORTS
// ============================================================================

export function detectIntent(message: string): Intent {
    const msgLower = message.toLowerCase();
    let detectedFeatures = detectFeatures(msgLower);
    let detectedHorizon = detectHorizonValue(msgLower);
    let detectedSite: number | null = null;

    const siteMatch = msgLower.match(SITE_PATTERN);
    if (siteMatch && siteMatch[1]) {
        detectedSite = parseInt(siteMatch[1], 10);
    }

    // Default to 24h if features are detected but no time is specified
    if (!detectedHorizon && detectedFeatures.length > 0) {
        detectedHorizon = 24;
    }

    // If no features detected but prediction is requested, default to all features
    const hasPredictionKeyword = ['dự báo', 'du bao', 'dự đoán', 'du doan', 'predict', 'forecast'].some(k => msgLower.includes(k));
    
    if (detectedFeatures.length === 0 && (detectedHorizon || hasPredictionKeyword)) {
        detectedFeatures = Object.keys(FEATURE_KEYWORDS);
        if (!detectedHorizon) detectedHorizon = 24;
    }

    return {
        is_prediction: detectedFeatures.length > 0 || hasPredictionKeyword,
        features: detectedFeatures,
        horizon: detectedHorizon,
        site: detectedSite,
        raw_query: message,
    };
}

export async function loadPredictions(
    feature: string, 
    horizon: number, 
    site: number = DEFAULT_SITE, 
    nLast?: number
): Promise<PredictionResult | null> {
    const n = nLast ?? horizon;
    
    const seriesPath = path.join(
        PROJECT_DIR, 
        'Proposed_Models', 
        feature, 
        'results', 
        `site_${site}`, 
        'series', 
        `series_SpikeDLinear_P${horizon}_${feature}.csv`
    );

    if (!fs.existsSync(seriesPath)) {
        return null;
    }

    try {
        const fileContent = fs.readFileSync(seriesPath, 'utf-8');
        
        // Use Papaparse with type assertions (Lint & Validate compliant)
        const parseResult = Papa.parse<CSVRow>(fileContent, {
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true, // converts strings to numbers automatically
        });

        const rows = parseResult.data;
        if (rows.length === 0) return null;

        const tail = rows.slice(-n);
        
        const predictedVals = tail.map(r => Number(r.Predicted)).filter(v => !isNaN(v));
        const actualVals = tail.map(r => Number(r.Actual)).filter(v => !isNaN(v));

        if (predictedVals.length === 0) return null;

        const mean = predictedVals.reduce((a, b) => a + b, 0) / predictedVals.length;
        const min = Math.min(...predictedVals);
        const max = Math.max(...predictedVals);
        const lastPred = predictedVals[predictedVals.length - 1];
        const lastAct = actualVals[actualVals.length - 1] ?? 0;

        return {
            feature,
            horizon,
            site,
            predicted_mean: mean,
            predicted_min: min,
            predicted_max: max,
            predicted_last: lastPred,
            actual_last: lastAct,
            n_points: predictedVals.length,
            predicted_values: predictedVals.map(v => Math.round(v * 100) / 100),
        };
    } catch (e) {
        console.error(`Error loading predictions for ${feature}:`, e);
        return null;
    }
}

export async function buildPredictionContext(
    features: string[], 
    horizon: number, 
    site: number = DEFAULT_SITE
): Promise<string> {
    const header = horizon >= 24 
        ? `[PREDICTION DATA — Site ${site}, Horizon ${horizon}h (${Math.floor(horizon / 24)} days)]`
        : `[PREDICTION DATA — Site ${site}, Horizon ${horizon}h]`;
        
    const lines: string[] = [header];
    const available: PredictionResult[] = [];
    const unavailable: string[] = [];

    for (const feature of features) {
        const result = await loadPredictions(feature, horizon, site);
        if (result) {
            available.push(result);
            lines.push(
                `\n  ${feature}:` +
                `\n    Predicted Mean: ${result.predicted_mean.toFixed(2)}` +
                `\n    Predicted Range: [${result.predicted_min.toFixed(2)} — ${result.predicted_max.toFixed(2)}]` +
                `\n    Latest Predicted: ${result.predicted_last.toFixed(2)}` +
                `\n    Latest Actual: ${result.actual_last.toFixed(2)}`
            );
        } else {
            unavailable.push(feature);
        }
    }

    if (unavailable.length > 0) {
        lines.push(`\n  ⚠️ No trained results available for: ${unavailable.join(', ')}`);
    }

    if (available.length > 0) {
        lines.push(
            `\n\n[INSTRUCTION] Based on the predicted values above, provide a comprehensive ` +
            `water quality assessment for a WATER TREATMENT PLANT (Nhà máy xử lý nước), NOT a river or natural ecosystem. ` +
            `Evaluate if the water is ready for the next treatment phase or safe for distribution. ` +
            `Referencing standard treatment thresholds (e.g., EC, pH 6.5-8.5, DO, Turbidity < 1 NTU for drinking, < 5 NTU max). ` +
            `Do NOT talk about fish or aquatic life. ` +
            `If any metric is concerning, explain what operational actions the plant manager should take ` +
            `(e.g., adjust aeration, add coagulants, backwash filters, modify pH with chemicals). ` +
            `Respond in Vietnamese. Format your response clearly with emojis, bold text, and bullet points.`
        );
    }

    return lines.join('\n');
}

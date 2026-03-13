import { loadPredictions } from './predictionLoader';

async function main() {
    console.log("Testing feature data loading...");
    for (const f of ['EC', 'pH', 'Temp', 'Flow', 'DO', 'Turbidity']) {
        const res = await loadPredictions(f, 24);
        console.log(f, res ? 'OK' : 'MISSING');
    }
}

main();
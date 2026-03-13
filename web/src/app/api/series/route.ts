import { NextResponse } from 'next/server';
import { getSeriesData } from '@/lib/dataLoader';

export async function GET(request: Request) {
    const { searchParams } = new URL(request.url);
    const site = searchParams.get('site') || '1463500';
    const target = searchParams.get('target') || 'EC';
    let horizon = searchParams.get('horizon') || '24h';
    
    // Remove "h" from horizon (e.g. "24h" -> "24")
    horizon = horizon.replace('h', '');

    try {
        const result = await getSeriesData(site, target, horizon);

        if (result.success) {
            return NextResponse.json(result);
        } else {
            return NextResponse.json({ error: result.error }, { status: 500 });
        }
    } catch (error: unknown) {
        return NextResponse.json({ error: error instanceof Error ? error.message : "Unknown error" }, { status: 500 });
    }
}
import { NextResponse } from 'next/server';
import { getRawData } from '@/lib/dataLoader';

export async function GET(request: Request) {
    const { searchParams } = new URL(request.url);
    const site = searchParams.get('site') || '1463500';
    const target = searchParams.get('target') || 'EC';

    try {
        const result = await getRawData(site, target);

        if (result.success) {
            return NextResponse.json(result);
        } else {
            return NextResponse.json({ error: result.error }, { status: 500 });
        }
    } catch (error: unknown) {
        return NextResponse.json({ error: error instanceof Error ? error.message : "Unknown error" }, { status: 500 });
    }
}

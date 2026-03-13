import dynamic from 'next/dynamic';

const StationMap = dynamic(() => import('./StationMap'), {
    ssr: false,
    loading: () => <div className="w-full h-full flex items-center justify-center bg-slate-100 rounded-lg text-slate-500 animate-pulse">Loading map...</div>
});

export default StationMap;

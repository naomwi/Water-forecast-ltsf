"use client";

import { MapContainer, TileLayer, Marker, Popup, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

// Fix for default marker icons in React Leaflet
const iconRetinaUrl = "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png";
const iconUrl = "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png";
const shadowUrl = "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png";

const defaultIcon = L.icon({
    iconRetinaUrl,
    iconUrl,
    shadowUrl,
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    tooltipAnchor: [16, -28],
    shadowSize: [41, 41]
});

L.Marker.prototype.options.icon = defaultIcon;

interface Station {
    id: string;
    name: string;
    lat: number;
    lng: number;
}

// Real stations for HydroPred
const MOCK_STATIONS: Station[] = [
    { id: "1463500", name: "USGS 01463500: Delaware River at Trenton, NJ", lat: 40.2216, lng: -74.7780 },
    { id: "1646500", name: "USGS 01646500: Potomac River near Washington, D.C.", lat: 38.9497, lng: -77.1276 },
    { id: "3216070", name: "USGS 03216070: Ohio River at Ironton, OH", lat: 38.5320, lng: -82.6859 },
    { id: "3321500", name: "USGS 03321500: Green River at Spottsville, KY", lat: 37.8583, lng: -87.4097 },
    { id: "14211720", name: "USGS 14211720: Willamette River at Portland, OR", lat: 45.5175, lng: -122.6691 }
];

export default function StationMap({
    selectedStation,
    onSelectStation,
}: {
    selectedStation: string;
    onSelectStation: (id: string) => void;
}) {
    return (
        <MapContainer
            center={[39.8283, -98.5795]}
            zoom={4}
            style={{ height: "100%", width: "100%", borderRadius: "0.5rem" }}
            className="z-0"
        >
            <TileLayer
                url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>'
            />
            {MOCK_STATIONS.map((station) => (
                <Marker
                    key={station.id}
                    position={[station.lat, station.lng]}
                    eventHandlers={{
                        click: () => onSelectStation(station.id),
                    }}
                >
                    <Tooltip direction="top" offset={[0, -20]} opacity={0.9}>
                        <span className="font-semibold">{station.id}</span>
                    </Tooltip>
                    <Popup>
                        <div className="font-semibold">{station.name}</div>
                        <div className="text-xs text-slate-500 text-center mt-1">ID: {station.id}</div>
                        {selectedStation === station.id && (
                            <div className="mt-2 text-xs font-bold text-blue-600 bg-blue-50 py-1 px-2 rounded w-full text-center">
                                Currently Selected
                            </div>
                        )}
                    </Popup>
                </Marker>
            ))}
        </MapContainer>
    );
}

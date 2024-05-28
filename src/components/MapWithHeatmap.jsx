import { useEffect, useRef, useState } from "react";
import PropTypes from "prop-types";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import "leaflet.heat";
import "./MapWithHeatmap.css";
import boatMarker from "../assets/boat_marker_512x512.png";
import { MapContainer, Marker, Popup, TileLayer } from "react-leaflet";
import { Stack } from "@mui/material";
import CustomButton from "./CustomButton";
import { RemoveCircle } from "@mui/icons-material";

const lerinsCenterPoint = [43.514185281889, 7.052543977186529];

const gradient = {
  0.6: "blue",
  0.7: "lime",
  0.8: "yellow",
  0.9: "orange",
  1: "red",
};

const gradientValues = ["1", "2-3", "4-6", "7-9", "10"];

const config = {
  // the minimum opacity the heat will start at
  minOpacity: 1,
  // zoom level where the points reach maximum intensity (as intensity scales with zoom), equals maxZoom of the map by default
  maxZoom: 21,
  // maximum point intensity, 1.0 by default
  max: 1,
  // radius of each "point" of the heatmap, 25 by default
  radius: 10,
  // amount of blur, 15 by default
  blur: 15,
  //color gradient config, e.g. {0.4: 'blue', 0.65: 'lime', 1: 'red'}
  gradient: gradient,
};

// Create a custom icon
const customIcon = new L.Icon({
  iconUrl: boatMarker,
  iconSize: [18, 18],
  iconAnchor: [9, 18],
});

const determineIntensity = (boatSize) => {
  // Define boat area ranges and their corresponding intensity values
  const areaRanges = [
    { min: 0, max: 10, intensity: 0.2 },
    { min: 11, max: 20, intensity: 0.4 },
    { min: 21, max: 30, intensity: 0.6 },
    { min: 31, max: 40, intensity: 0.8 },
    { min: 41, max: Infinity, intensity: 1 },
  ];

  // Find the appropriate intensity value based on boat size
  for (const range of areaRanges) {
    if (boatSize >= range.min && boatSize <= range.max) {
      return range.intensity;
    }
  }

  
  return 0.1; // Default intensity value
};

export default function MapWithHeatmap(props) {
  const { heatmapData, isMarker, setData } = props;
  const mapRef = useRef(null);
  const [markers, setMarkers] = useState([]);
  const [previousHeatmapData, setPreviousHeatmapData] = useState([]);
  const [heatLayers, setHeatLayers] = useState([]);
  const [isHeatmap, setIsHeatmap] = useState(false);
  const [isLegend, setIsLegend] = useState(false);

  const legend = L.control({ position: "bottomright" });

  const handleRemoveMapData = (removeData = true) => {
    setMarkers([]);
    setIsHeatmap(false);
    if (removeData) {
      setData([]);
    }
    heatLayers.forEach((layer) => {
      layer.removeFrom(mapRef.current);
    });
  };
  
  const handleHeatmap = (map) => {
    if (!isHeatmap || previousHeatmapData !== heatmapData) {
      // Remove previous data
      handleRemoveMapData(false);
  
      // Filter heatmap data based on boat area > 100
      const filteredHeatmapData = heatmapData.filter(point =>  point.area > 74);
  
      // Extracting data to leaflet.heat format
      const heatmapDataForLeaflet = filteredHeatmapData.map((point, key) => {
        // Create a unique key
        const uniqueKey = `${key}-${point.lat}-${point.lng}-${point.date}`;
  
        // Extracting image name
        const image_path = point.image.split("/");
        const image = image_path[image_path.length - 1];
  
        // Getting markers
        const marker = (
          <Marker
            key={uniqueKey}
            icon={customIcon}
            position={[point.lat, point.lng]}
          >
            <Popup>{`${point.date} - ${image} \n ${point.lat} ${point.lng}`}</Popup>
          </Marker>
        );
        setMarkers((markers) => [...markers, marker]);
  
        // Determine intensity based on boat size
        const intensity = determineIntensity(point.area);
  
        return {
          lat: point.lat,
          lon: point.lng,
          value: intensity,
        };
      });
  
      // Create the heatmap layer on the map
      const heatLayer = L.heatLayer(heatmapDataForLeaflet, config );
      setHeatLayers([heatLayer]);
  
      heatLayer.addTo(map);
      setIsHeatmap(true);
      setPreviousHeatmapData(heatmapData);
    }
  };

  const handleLegend = (map) => {
    // if the map don got a legend, add it
    if (!isLegend) {
      legend.onAdd = function (map) {
        const div = L.DomUtil.create("div", "info legend");

        const sortedKeys = Object.keys(gradient).sort((a, b) => a > b);

        let suffix = "";

        div.innerHTML += `<div>Number of boats</div>`;

        // Create the gradient legend
        for (let [index, key] of sortedKeys.entries()) {
          const color = gradient[key];
          //if it is last elemment, add a + at the end
          if (key == sortedKeys[sortedKeys.length - 1]) {
            suffix = "+";
          }
          div.innerHTML += `<div><i style="background:${color}"></i> ${gradientValues[index]}${suffix}</div>`;
        }

        return div;
      };

      legend.addTo(map);
      setIsLegend(true);
    }
  };

  useEffect(() => {
    if (mapRef.current) {
      const map = mapRef.current;
      
      if (heatmapData.length > 0) {
        // Add the heatmap
        handleHeatmap(map);

        // Add the legend
        handleLegend(map);
      }
    }
  }, [heatmapData, isMarker]);

  return (
    <Stack
      spacing={1}
      sx={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <MapContainer
        ref={mapRef}
        center={lerinsCenterPoint}
        zoom={15}
        scrollWheelZoom={true}
        id="map-content"
        style={{ height: "50vh", width: "80vw" }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {isMarker && markers}
      </MapContainer>
      <CustomButton
        id="remove-map-data"
        text="Remove Map Data"
        icon={<RemoveCircle />}
        color="error"
        onClick={handleRemoveMapData}
      />
    </Stack>
  );
}

MapWithHeatmap.propTypes = {
  heatmapData: PropTypes.array,
  isMarker: PropTypes.bool,
  setData: PropTypes.func.isRequired,
};

MapWithHeatmap.defaultProps = {
  heatmapData: [],
  isMarker: false,
};

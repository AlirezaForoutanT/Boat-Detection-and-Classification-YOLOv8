import { Paper, Stack, ToggleButton, Typography } from "@mui/material";
import RoomIcon from '@mui/icons-material/Room';
import PropTypes from "prop-types";
import "./MapFilters.css";
import MapSlider from "./MapSlider";

export default function MapFilters(props) {

  const {isMarker, setIsMarker, originalData, setHeatmapData} = props;

  return (
    <Paper sx={{
      padding: 1,
      textAlign: "left",
    }}>
      <Typography variant="h6">Filters</Typography>
      <Stack direction="column">
        <Stack direction="row">
          <Typography className="filter-stack-typography">Toggle markers</Typography>
          <ToggleButton
            value="check"
            selected={isMarker}
            onChange={() => {
              setIsMarker(!isMarker);
            }}
          >
            <RoomIcon/>
          </ToggleButton>
        </Stack>
        <Stack direction="row">
          {/* Range slider to display heatmap within the range */}
          <Typography className="filter-stack-typography">Datetime range</Typography>
          <MapSlider originalData={originalData} setHeatmapData={setHeatmapData}/>
        </Stack>
      </Stack>
    </Paper>
  )
}

MapFilters.propTypes = {
  isMarker: PropTypes.bool.isRequired,
  setIsMarker: PropTypes.func.isRequired,
  originalData: PropTypes.array.isRequired,
  setHeatmapData: PropTypes.func.isRequired,
};

MapFilters.defaultProps = {
};


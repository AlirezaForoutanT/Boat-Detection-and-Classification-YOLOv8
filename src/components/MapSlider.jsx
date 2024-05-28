import { Box, Slider, Stack, Typography } from "@mui/material";
import { useEffect, useState } from "react";
import PropTypes from "prop-types";
import moment from "moment";

const dateFormat = "YYYY-MM-DD HH:mm:ss";
const minRange = 1;
const defaultSliderRange = [-1, 10];

function getTimestamp(date) {
  return moment(date, dateFormat).valueOf();
}


export default function MapSlider(props) {
  const { originalData, setHeatmapData } = props;

  const [value, setValue] = useState(defaultSliderRange);
  const [uniqueDates, setUniqueDates] = useState([]);
  const [nbBoats, setNbBoats] = useState(0);

  const handleUniqueDates = (data) => {
    // extract unique dates from the data
    const uniqueDatesList = [...new Set(data.map((data) => data.date))];

    // set default value to the slider if the slider is not set
    if (value == defaultSliderRange) {
      setValue([0, uniqueDatesList.length - 1]);
    }

    // set the unique dates to the state
    setUniqueDates(uniqueDatesList);
    setNbBoats(data.length);
  };

  const handleDataRange = (range) => {
    if (uniqueDates.length === 0) return;
    // filter the data to get only the data within the range
    const filteredData = originalData.filter(
      (data) =>
        getTimestamp(data.date) >= getTimestamp(uniqueDates[range[0]]) &&
        getTimestamp(data.date) <= getTimestamp(uniqueDates[range[1]])
    );

    // set the filtered data to the heatmap
    setHeatmapData(filteredData);
    setNbBoats(filteredData.length);
  };

  const handleChange = (event, newValue, activeThumb) => {
    // set new value and respect the minRange
    if (activeThumb === 0) {
      setValue([Math.min(newValue[0], value[1] - minRange), value[1]]);
    } else {
      setValue([value[0], Math.max(newValue[1], value[0] + minRange)]);
    }
  };

  const buildMarks = () => {
    if (uniqueDates.length === 0) return [];
    let marks = [
      {
        value: 0,
        label: uniqueDates[0],
      },
      {
        value: uniqueDates.length - 1,
        label: uniqueDates[uniqueDates.length - 1],
      },
    ];
    return marks;
  };

  // get the text to display on the slider
  const valueText = (value) => {
    return uniqueDates[value];
  };

  useEffect(() => {
    if (originalData.length !== 0 && uniqueDates.length === 0) {
      handleUniqueDates(originalData);
    }

    else if (originalData.length === 0 && uniqueDates.length !== 0) {
      handleDataRange(defaultSliderRange);
      setUniqueDates([]);
    }
  }, [originalData]);

  useEffect(() => {
    handleDataRange(value);
  }, [value]);

  return (
    <Box sx={{ width: "35vw"}}>
      <Box
        sx={{ display: "flex", justifyContent: "center", alignItems: "center" }}
      >
        <Typography>{`${nbBoats} boat${nbBoats > 1 ? "s" : ""}`}</Typography>
      </Box>
      <Slider
        getAriaLabel={() => "Date range"}
        value={value}
        onChange={handleChange}
        valueLabelFormat={valueText}
        valueLabelDisplay="auto"
        getAriaValueText={valueText}
        min={uniqueDates.length > 0 ? 0 : -1}
        max={uniqueDates.length > 0 ? uniqueDates.length - 1 : 10}
        step={1}
        disableSwap={true}
        marks={buildMarks()}
        disabled={uniqueDates.length === 0}
      />
    </Box>
  );
}

MapSlider.propTypes = {
  originalData: PropTypes.array.isRequired,
  setHeatmapData: PropTypes.func.isRequired,
};

MapSlider.defaultProps = {};





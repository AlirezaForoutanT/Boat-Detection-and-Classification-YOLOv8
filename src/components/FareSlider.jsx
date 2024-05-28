import { Box, Slider, Typography } from "@mui/material";
import { useState } from "react";

function FareSlider({ min = 10, max = 200, onFareChange }) {
  const [fare, setFare] = useState(min);

  const handleChange = (event, newValue) => {
    setFare(newValue);
    if (onFareChange) {
      onFareChange(newValue);
    }
  };

  return (
    <Box sx={{ width: '100%', padding: '20px' }}>
      <Typography id="fare-slider-label" gutterBottom>
        Boat Fare Range
      </Typography>
      <Slider
        value={fare}
        onChange={handleChange}
        aria-labelledby="fare-slider-label"
        valueLabelDisplay="auto"
        min={min}
        max={max}
      />
    </Box>
  );
}

export default FareSlider;
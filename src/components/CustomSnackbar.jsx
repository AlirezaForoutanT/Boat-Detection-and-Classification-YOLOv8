import { useState, useEffect } from "react";
import Stack from "@mui/material/Stack";
import Snackbar from "@mui/material/Snackbar";
import Alert from "@mui/material/Alert";
import PropTypes from "prop-types";

export default function CustomSnackbar({
  id,
  open,
  message,
  severity,
  color,
  autoHideDuration,
  onClose,
}) {
  const [openSnackBar, setOpenSnackBar] = useState(open);
  const [mySeverity, setMySeverity] = useState(severity);
  const [customColor, setCustomColor] = useState(color);

  useEffect(() => {
    setOpenSnackBar(open);
    setMySeverity(severity);
    if (color === null) {
      setCustomColor(severity);
    } else {
      setCustomColor(color);
    }
  }, [open, severity, color]);

  const handleClose = (event, reason) => {
    if (reason === "clickaway") {
      return;
    }
    setOpenSnackBar(false);
    if (onClose) {
      onClose();
    }
  };

  return (
    <Stack spacing={2} sx={{ width: "50%" }}>
      <Snackbar
        open={openSnackBar}
        autoHideDuration={autoHideDuration}
        onClose={handleClose}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          id={`${id}-snackbar`}
          elevation={6}
          variant="filled"
          onClose={handleClose}
          severity={mySeverity}
          color={customColor}
          sx={{ width: "100%" }}
        >
          {message}
        </Alert>
      </Snackbar>
    </Stack>
  );
}

CustomSnackbar.propTypes = {
  autoHideDuration: PropTypes.number,
  color: PropTypes.string,
  id: PropTypes.string.isRequired,
  message: PropTypes.string.isRequired,
  open: PropTypes.bool,
  severity: PropTypes.string,
  onClose: PropTypes.func,
};

CustomSnackbar.defaultProps = {
  open: false,
  severity: "info",
  color: null,
  autoHideDuration: 5000,
  onClose: () => {},
};

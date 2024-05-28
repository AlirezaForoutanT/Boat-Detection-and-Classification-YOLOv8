import { useState, useEffect } from "react";
import { Collapse, Paper, Stack, Typography } from "@mui/material";
import "./App.css";
import MapWithHeatmap from "./components/MapWithHeatmap";
import CustomDropzone from "./components/CustomDropzone";
import CustomButton from "./components/CustomButton";
import getCsvToJsonData from "./utils/getCsvToJsonData";
import {
  Cancel,
  CheckCircle,
  ExpandLess,
  ExpandMore,
} from "@mui/icons-material";
import CustomSnackbar from "./components/CustomSnackBar";
import MapFilters from "./components/MapFilters";
import FareSlider from './components/FareSlider.jsx'; // The new Fare Slider component


export default function App() {
  const [data, setData] = useState([]);
  const [heatmapData, setHeatmapData] = useState([]);
  const [csv, setCsv] = useState([]);
  const [csvDropzone, setCsvDropzone] = useState([]);
  const [successMessage, setSuccessMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [open, setOpen] = useState(false);
  const [isMarker, setIsMarker] = useState(false);

  const handleFareChange = (newFare) => {
    console.log("New fare selected:", newFare)};
    
  const handleCollapseClick = () => {
    setOpen(!open);
  };

  const handleCsvImport = async () => {
    try {
      const response = await getCsvToJsonData(csv);
      console.log("response \n", response);
      setData(response);

      setSuccessMessage("File(s) have been uploaded");
    } catch (err) {
      setErrorMessage(
        "An error occured while reading the given file(s) : ",
        err
      );
    }
  };

  useEffect(() => {
    const dropzone = (
      <Stack>
        <CustomDropzone
          accept={{
            "text/csv": [],
            "application/vnd.ms-excel": [],
          }}
          maxFiles={5}
          setFiles={setCsv}
          clearFileList={csv.length === 0}
        />
        <Stack>
          <Typography>Files must follow this format :</Typography>
          <Typography variant="caption">
            class,lat,lng,date
            <br />
            0,43.51090306995201,7.036932757649052,2023-09-15 15:36:01
            <br />
            0,43.51090308381105,7.036932763102577,2023-09-15 15:36:01
            <br />
            0,43.51090314635359,7.036932920279365,2023-09-15 15:36:01
          </Typography>
        </Stack>
      </Stack>
    );
    setCsvDropzone(dropzone);
  }, [csv]);

  useEffect(() => {
    setHeatmapData(data);
  }, [data]);

  return (
    <>
      <main>


        <h1 className="title">Mooring Project</h1>

        <p className="description">Boat detection heatmap</p>
        <Paper
          elevation={0}
          sx={{
            display: { xs: "flex" },
            alignItems: "center",
            justifyContent: "space-between",
            paddingX: 2,
            cursor: "pointer",
          }}
          onClick={handleCollapseClick}
        >
          <Typography variant="overline">Upload CSV files</Typography>
          {open ? <ExpandLess /> : <ExpandMore />}
        </Paper>
        <Collapse in={open}>
          <Stack sx={{ margin: 2 }}>
            {csvDropzone}
            <Stack
              sx={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-evenly",
                flexDirection: "row",
                marginY: "20px",
              }}
            >
              <CustomButton
                id="validate-csv-import"
                text="Validate"
                icon={<CheckCircle />}
                color="primary"
                disabled={csv.length === 0}
                onClick={handleCsvImport}
              />
              <CustomButton
                id="cancel-csv-import"
                text="Cancel"
                icon={<Cancel />}
                color="secondary"
                onClick={() => {
                  setCsv([]);
                }}
              />
            </Stack>
          </Stack>
        </Collapse>
        <MapWithHeatmap heatmapData={heatmapData} isMarker={isMarker} setData={setData} />
        <MapFilters isMarker={isMarker} setIsMarker={setIsMarker} originalData={data} setHeatmapData={setHeatmapData}/>
      </main>
                
      <CustomSnackbar
        id="dropzone-success"
        open={successMessage !== ""}
        message={successMessage}
        severity="success"
        onClose={() => {
          setSuccessMessage("");
        }}
      />
      <CustomSnackbar
        id="dropzone-error"
        open={errorMessage !== ""}
        message={errorMessage}
        severity="error"
        onClose={() => {
          setErrorMessage("");
        }}
      />
      <div>
        <FareSlider onFareChange={handleFareChange} />
      </div>
      <footer>
        {
          "Université Côte d'Azur."
        }
      </footer>
    </>
  );
}

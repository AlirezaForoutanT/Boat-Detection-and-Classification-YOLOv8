import { List, ListItem, Paper, Typography, useTheme } from "@mui/material";
import Dropzone from "react-dropzone";
import { styled } from "@mui/material/styles";
import { FileUpload } from "@mui/icons-material";
import { useCallback, useState, useEffect } from "react";
import PropTypes from "prop-types";
import CustomSnackbar from "./CustomSnackBar";

/**
 * Dropzone component which allow to upload files
 * @param {Object} props Component parameters
 * @param {Function} props.setFiles React setState of file array
 * @param {Number} props.maxFiles Maximum number of files to upload
 * @returns
 */
export default function CustomDropzone(props) {
  const { setFiles, maxFiles, clearFileList } = props;

  const theme = useTheme();
  const [acceptedFilesList, setAcceptedFilesList] = useState([]);
  const [errorMessage, setErrorMessage] = useState("");

  const DropzonePaper = styled(Paper)({
    borderWidth: 1,
    backgroundColor: theme.palette.secondary.light,
    opacity: 0.75,
    color: theme.palette.secondary.contrastText,
    cursor: "pointer",
  });

  const handleOnDropAccepted = useCallback((acceptedFiles) => {
    const acceptedFileItems = acceptedFiles.map((file, index) => {
      const reader = new FileReader();

      reader.onabort = () => console.warn("File reading was aborted");
      reader.onerror = () => console.warn("File reading has failed");
      reader.onload = () => {
        // Récupération du contenu des fichiers
        const fileObject = {
          file,
          buffer: reader.result,
        };
        setFiles((prevFiles) => [...prevFiles, fileObject]);
      };
      // reader.readAsArrayBuffer(file);
      reader.readAsText(file);

      return (
        <ListItem key={index}>
          {file.name} - {file.size} bytes
        </ListItem>
      );
    });
    setAcceptedFilesList(acceptedFileItems);
  }, []);

  const handleOnDropRejected = (rejectedFiles) => {
    if (rejectedFiles.length > maxFiles) {
      setErrorMessage(`You can't upload more than ${maxFiles} file(s) !`);
    } else {
      setErrorMessage(
        "Given file(s) was/were not uploaded, verify the file extension."
      );
    }
  };

  useEffect(() => {
    if (clearFileList) {
      setAcceptedFilesList([]);
      setFiles([]);
    }
  }, [clearFileList, setFiles]);

  return (
    <>
      <Dropzone
        onDropAccepted={handleOnDropAccepted}
        onDropRejected={handleOnDropRejected}
        {...props}
      >
        {({ getRootProps, getInputProps }) => (
          <DropzonePaper
            {...getRootProps()}
            sx={{
              padding: 3,
              marginBottom: 1,
            }}
          >
            <input {...getInputProps()} />
            <Typography
              sx={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <FileUpload sx={{ marginRight: 1 }} />
              Drag, drop or click to import your files
            </Typography>
          </DropzonePaper>
        )}
      </Dropzone>
      <List>{acceptedFilesList}</List>
      <CustomSnackbar
        id="dropzone-error"
        open={errorMessage !== ""}
        message={errorMessage}
        severity="error"
        onClose={() => {
          setErrorMessage("");
        }}
      />
    </>
  );
}

CustomDropzone.propTypes = {
  maxFiles: PropTypes.number.isRequired,
  setFiles: PropTypes.func.isRequired,
  clearFileList: PropTypes.bool,
};
CustomDropzone.defaultProps = {
  clearFileList: false,
};

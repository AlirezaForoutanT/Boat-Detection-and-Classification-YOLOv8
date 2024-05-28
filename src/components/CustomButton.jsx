import PropTypes from "prop-types";
import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import Button from "@mui/material/Button";
import { useEffect, useState } from "react";
import { Tooltip } from "@mui/material";

/**
 * Button component
 * @param {Object} props Component parameters
 * @param {String} props.id Button identifier
 * @param {String} props.text Text displayed on the button
 * @param {Node} props.icon Icon place in front of the text
 * @param {String} props.color Button color
 * @param {String} props.variant Button variant display
 * @param {Boolean} props.isResponsive Display the button in responsive mode if true or full if false
 * @param {Function} props.onClick Function executed when the button is clicked
 * @param {Boolean} props.show Display or not the button
 * @param {String} props.size Button size
 * @param {Object} props.sx CSS properties
 * @param {Boolean} props.onlyIcon Only display the button in icon form
 * @returns
 */
export default function CustomButton({
  id,
  text,
  icon,
  color,
  variant,
  isResponsive,
  onlyIcon,
  onClick,
  show,
  size,
  sx,
  disabled,
}) {
  const [showButton, setShowButton] = useState(show);
  const [myIcon, setMyIcon] = useState(null);
  const [myButton, setMyButton] = useState(null);

  useEffect(() => {
    const iconButton = (
      <IconButton
        id={`${id}-icon-button`}
        size={size}
        onClick={onClick}
        color={color}
        disabled={disabled}
      >
        {icon}
      </IconButton>
    );

    const newIcon = disabled ? (
      iconButton
    ) : (
      <Tooltip title={text}>{iconButton}</Tooltip>
    );

    const newButton = (
      <Button
        id={`${id}-button`}
        color={color}
        variant={variant}
        onClick={onClick}
        startIcon={icon}
        size={size}
        sx={sx}
        disabled={disabled}
      >
        {text}
      </Button>
    );

    setShowButton(show);
    setMyIcon(newIcon);
    setMyButton(newButton);
  }, [text, icon, color, variant, isResponsive, onlyIcon, onClick, show, size, sx]);

  return (
    <Box>
      {showButton && (
        <Box>
          {onlyIcon ? (
            <Box sx={{ display: { xs: "flex", md: "flex" } }}>{myIcon}</Box>
          ) : (
            <Box>
              {isResponsive ? (
                <Box>
                  <Box sx={{ display: { xs: "flex", md: "none" } }}>
                    {myIcon}
                  </Box>
                  <Box sx={{ display: { xs: "none", md: "flex" } }}>
                    {myButton}
                  </Box>
                </Box>
              ) : (
                <Box sx={{ display: { xs: "flex", md: "flex" } }}>
                  {myButton}
                </Box>
              )}
            </Box>
          )}
        </Box>
      )}
    </Box>
  );
}

CustomButton.propTypes = {
  color: PropTypes.string,
  disabled: PropTypes.bool,
  icon: PropTypes.node,
  id: PropTypes.string.isRequired,
  isResponsive: PropTypes.bool,
  onlyIcon: PropTypes.bool,
  show: PropTypes.bool,
  size: PropTypes.string,
  sx: PropTypes.object,
  text: PropTypes.string.isRequired,
  variant: PropTypes.string,
  onClick: PropTypes.func,
};

CustomButton.defaultProps = {
  color: "primary",
  disabled: false,
  icon: null,
  isResponsive: true,
  onlyIcon: false,
  show: true,
  size: "large",
  sx: {},
  variant: "contained",
  onClick: () => {},
};

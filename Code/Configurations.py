

#----------------------------------------------------------------
class ConfigSection:
    """Represents a section in the configuration file."""
    def __init__(self, name, settings, defaults):
        self._name = name
        self._settings = settings
        self._defaults = defaults

    def __getattr__(self, key):
        """Retrieve a value from the section with fallback to defaults."""
        if key in self._settings:
            return self._cast_value(self._settings[key])
        if key in self._defaults:
            return self._cast_value(self._defaults[key])
        raise AttributeError(f"'{self._name}' section has no key '{key}'")

    def __setattr__(self, key, value):
        """Set a value in the section."""
        if key in ("_name", "_settings", "_defaults"):
            super().__setattr__(key, value)
        else:
            self._settings[key] = str(value)  # Store all values as strings in the config file

    def __delattr__(self, key):
        """Delete a key from the section."""
        if key in self._settings:
            del self._settings[key]
        else:
            raise AttributeError(f"'{self._name}' section has no key '{key}'")

    def __repr__(self):
        """String representation of the section."""
        return f"<ConfigSection '{self._name}': {self._settings}>"

    @staticmethod
    def _cast_value(value):
        """Attempt to cast a string value to its appropriate type."""
        if value.isdigit():  # Check if the value contains only digits
            return int(value)
        try:
            return float(value)  # Attempt to cast to float if it's a numeric string
        except ValueError:
            pass
        try:
            return eval(value)
        except ValueError:
            return value  # Return as string if it can't be cast


#----------------------------------------------------------------
class ConfigManager:
    """A class to manage configurations with default values."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.configs = {}
        self.defaults = {}
        self._load_configs()

    def _load_configs(self):
        """Load configurations from the file."""
        current_section = None
        try:
            with open(self.file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('#'):  # Skip empty lines and comments
                        continue
                    if line.startswith('[') and line.endswith(']'):  # New section
                        current_section = line[1:-1]
                        if current_section == "defaults":
                            self.defaults = {}
                        else:
                            self.configs[current_section] = {}
                    elif '=' in line and current_section:
                        key, value = line.split('=', 1)
                        key, value = key.strip(), value.strip()
                        if current_section == "defaults":
                            self.defaults[key] = value
                        else:
                            self.configs[current_section][key] = value
        except FileNotFoundError:
            # Initialize defaults if file doesn't exist
            self.defaults = {}

    def _save_configs(self):
        """Save the current configurations to the file."""
        with open(self.file_path, 'w') as file:
            # Save defaults
            file.write("[defaults]\n")
            for key, value in self.defaults.items():
                file.write(f"{key}={value}\n")
            file.write("\n")

            # Save configurations
            for section, settings in self.configs.items():
                file.write(f"[{section}]\n")
                for key, value in settings.items():
                    file.write(f"{key}={value}\n")
                file.write("\n")

    def __getattr__(self, section):
        """Provide access to sections as attributes."""
        if section in self.configs:
            return ConfigSection(section, self.configs[section], self.defaults)
        if section == "defaults":
            return ConfigSection("defaults", self.defaults, {})
        raise AttributeError(f"Configuration section '{section}' does not exist.")

    def add_section(self, section: str):
        """Add a new section if it doesn't already exist."""
        if section not in self.configs:
            self.configs[section] = {}
            self._save_configs()

    def delete_section(self, section: str):
        """Delete a section and its settings."""
        if section in self.configs:
            del self.configs[section]
            self._save_configs()

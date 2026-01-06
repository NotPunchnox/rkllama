"""Tests for config schema module."""

import pytest

from rkllama.config.config_schema import (
    ConfigField,
    ConfigSchema,
    ConfigSectionSchema,
    FieldType,
    create_rkllama_schema,
)


class TestFieldType:
    """Tests for FieldType enum."""

    def test_field_types_exist(self):
        """Test all expected field types exist."""
        assert FieldType.STRING.value == "string"
        assert FieldType.INTEGER.value == "integer"
        assert FieldType.FLOAT.value == "float"
        assert FieldType.BOOLEAN.value == "boolean"
        assert FieldType.LIST.value == "list"
        assert FieldType.PATH.value == "path"


class TestConfigField:
    """Tests for ConfigField class."""

    # =========================================================================
    # String Field Tests
    # =========================================================================

    def test_string_field_default(self):
        """Test string field with default value."""
        field = ConfigField(FieldType.STRING, "default_value")
        assert field.validate(None) == "default_value"

    def test_string_field_conversion(self):
        """Test string field converts various types."""
        field = ConfigField(FieldType.STRING, "")
        assert field.validate("hello") == "hello"
        assert field.validate(123) == "123"
        assert field.validate(45.6) == "45.6"
        assert field.validate(True) == "True"

    def test_string_field_with_options(self):
        """Test string field with allowed options."""
        field = ConfigField(FieldType.STRING, "a", options=["a", "b", "c"])
        assert field.validate("a") == "a"
        assert field.validate("b") == "b"

    def test_string_field_invalid_option(self):
        """Test string field rejects invalid options."""
        field = ConfigField(FieldType.STRING, "a", options=["a", "b", "c"])
        with pytest.raises(ValueError, match="not in allowed options"):
            field.validate("d")

    # =========================================================================
    # Integer Field Tests
    # =========================================================================

    def test_integer_field_default(self):
        """Test integer field with default value."""
        field = ConfigField(FieldType.INTEGER, 42)
        assert field.validate(None) == 42

    def test_integer_field_conversion(self):
        """Test integer field converts strings."""
        field = ConfigField(FieldType.INTEGER, 0)
        assert field.validate("123") == 123
        assert field.validate(456) == 456
        assert field.validate(78.9) == 78  # truncates

    def test_integer_field_min_value(self):
        """Test integer field with minimum value."""
        field = ConfigField(FieldType.INTEGER, 10, min_value=5)
        assert field.validate(10) == 10
        assert field.validate(5) == 5

    def test_integer_field_min_value_violation(self):
        """Test integer field rejects values below minimum."""
        field = ConfigField(FieldType.INTEGER, 10, min_value=5)
        with pytest.raises(ValueError, match="less than minimum"):
            field.validate(4)

    def test_integer_field_max_value(self):
        """Test integer field with maximum value."""
        field = ConfigField(FieldType.INTEGER, 10, max_value=100)
        assert field.validate(50) == 50
        assert field.validate(100) == 100

    def test_integer_field_max_value_violation(self):
        """Test integer field rejects values above maximum."""
        field = ConfigField(FieldType.INTEGER, 10, max_value=100)
        with pytest.raises(ValueError, match="greater than maximum"):
            field.validate(101)

    def test_integer_field_range(self):
        """Test integer field with both min and max."""
        field = ConfigField(FieldType.INTEGER, 50, min_value=1, max_value=100)
        assert field.validate(1) == 1
        assert field.validate(50) == 50
        assert field.validate(100) == 100

    def test_integer_field_invalid_string(self):
        """Test integer field rejects non-numeric strings."""
        field = ConfigField(FieldType.INTEGER, 0)
        with pytest.raises(ValueError, match="Failed to convert"):
            field.validate("not_a_number")

    # =========================================================================
    # Float Field Tests
    # =========================================================================

    def test_float_field_default(self):
        """Test float field with default value."""
        field = ConfigField(FieldType.FLOAT, 3.14)
        assert field.validate(None) == 3.14

    def test_float_field_conversion(self):
        """Test float field converts strings and integers."""
        field = ConfigField(FieldType.FLOAT, 0.0)
        assert field.validate("3.14") == 3.14
        assert field.validate(42) == 42.0
        assert field.validate(2.718) == 2.718

    def test_float_field_min_value(self):
        """Test float field with minimum value."""
        field = ConfigField(FieldType.FLOAT, 0.5, min_value=0.0)
        assert field.validate(0.0) == 0.0
        assert field.validate(0.5) == 0.5

    def test_float_field_min_value_violation(self):
        """Test float field rejects values below minimum."""
        field = ConfigField(FieldType.FLOAT, 0.5, min_value=0.0)
        with pytest.raises(ValueError, match="less than minimum"):
            field.validate(-0.1)

    def test_float_field_max_value(self):
        """Test float field with maximum value."""
        field = ConfigField(FieldType.FLOAT, 0.5, max_value=1.0)
        assert field.validate(0.5) == 0.5
        assert field.validate(1.0) == 1.0

    def test_float_field_max_value_violation(self):
        """Test float field rejects values above maximum."""
        field = ConfigField(FieldType.FLOAT, 0.5, max_value=1.0)
        with pytest.raises(ValueError, match="greater than maximum"):
            field.validate(1.1)

    def test_float_field_invalid_string(self):
        """Test float field rejects non-numeric strings."""
        field = ConfigField(FieldType.FLOAT, 0.0)
        with pytest.raises(ValueError, match="Failed to convert"):
            field.validate("not_a_number")

    # =========================================================================
    # Boolean Field Tests
    # =========================================================================

    def test_boolean_field_default(self):
        """Test boolean field with default value."""
        field = ConfigField(FieldType.BOOLEAN, True)
        assert field.validate(None) is True

        field = ConfigField(FieldType.BOOLEAN, False)
        assert field.validate(None) is False

    def test_boolean_field_true_strings(self):
        """Test boolean field recognizes true strings."""
        field = ConfigField(FieldType.BOOLEAN, False)
        for val in ["true", "True", "TRUE", "yes", "Yes", "YES", "1", "on", "ON", "y", "Y"]:
            assert field.validate(val) is True

    def test_boolean_field_false_strings(self):
        """Test boolean field recognizes false strings."""
        field = ConfigField(FieldType.BOOLEAN, True)
        for val in ["false", "False", "FALSE", "no", "No", "NO", "0", "off", "OFF"]:
            assert field.validate(val) is False

    def test_boolean_field_from_bool(self):
        """Test boolean field accepts boolean values directly."""
        field = ConfigField(FieldType.BOOLEAN, False)
        assert field.validate(True) is True
        assert field.validate(False) is False

    def test_boolean_field_from_int(self):
        """Test boolean field converts integers to boolean."""
        field = ConfigField(FieldType.BOOLEAN, False)
        assert field.validate(1) is True
        assert field.validate(0) is False

    # =========================================================================
    # List Field Tests
    # =========================================================================

    def test_list_field_default(self):
        """Test list field with default value."""
        field = ConfigField(FieldType.LIST, ["a", "b"])
        assert field.validate(None) == ["a", "b"]

    def test_list_field_from_string(self):
        """Test list field parses comma-separated strings."""
        field = ConfigField(FieldType.LIST, [])
        assert field.validate("a,b,c") == ["a", "b", "c"]
        assert field.validate("one, two, three") == ["one", "two", "three"]

    def test_list_field_from_list(self):
        """Test list field accepts lists directly."""
        field = ConfigField(FieldType.LIST, [])
        assert field.validate(["x", "y", "z"]) == ["x", "y", "z"]

    def test_list_field_empty_string(self):
        """Test list field handles empty strings."""
        field = ConfigField(FieldType.LIST, [])
        assert field.validate("") == []

    def test_list_field_with_item_type(self):
        """Test list field converts items to specified type."""
        field = ConfigField(FieldType.LIST, [], item_type=FieldType.INTEGER)
        assert field.validate("1,2,3") == [1, 2, 3]

    def test_list_field_invalid_type(self):
        """Test list field rejects invalid types."""
        field = ConfigField(FieldType.LIST, [])
        with pytest.raises(ValueError, match="Cannot convert"):
            field.validate(12345)

    # =========================================================================
    # Path Field Tests
    # =========================================================================

    def test_path_field_default(self):
        """Test path field with default value."""
        field = ConfigField(FieldType.PATH, "/default/path")
        assert field.validate(None) == "/default/path"

    def test_path_field_conversion(self):
        """Test path field converts values to strings."""
        field = ConfigField(FieldType.PATH, "")
        assert field.validate("/some/path") == "/some/path"
        assert field.validate("relative/path") == "relative/path"

    # =========================================================================
    # Required Field Tests
    # =========================================================================

    def test_required_field_with_value(self):
        """Test required field accepts values."""
        field = ConfigField(FieldType.STRING, "", required=True)
        assert field.validate("value") == "value"

    def test_required_field_without_value(self):
        """Test required field rejects None."""
        field = ConfigField(FieldType.STRING, "", required=True)
        with pytest.raises(ValueError, match="required"):
            field.validate(None)


class TestConfigSectionSchema:
    """Tests for ConfigSectionSchema class."""

    def test_section_creation(self):
        """Test section creation with description."""
        section = ConfigSectionSchema("Test section")
        assert section.description == "Test section"
        assert section.fields == {}

    def test_add_field(self):
        """Test adding a field to section."""
        section = ConfigSectionSchema()
        field = ConfigField(FieldType.STRING, "default")
        section.add_field("test_field", field)
        assert "test_field" in section.fields

    def test_string_helper(self):
        """Test string() helper method."""
        section = ConfigSectionSchema()
        result = section.string("name", "default", "Description")
        assert result is section  # Returns self for chaining
        assert "name" in section.fields
        assert section.fields["name"].field_type == FieldType.STRING

    def test_integer_helper(self):
        """Test integer() helper method."""
        section = ConfigSectionSchema()
        section.integer("port", 8080, "Port number", min_value=1, max_value=65535)
        assert section.fields["port"].field_type == FieldType.INTEGER
        assert section.fields["port"].min_value == 1
        assert section.fields["port"].max_value == 65535

    def test_float_helper(self):
        """Test float() helper method."""
        section = ConfigSectionSchema()
        section.float("temperature", 0.7, "Temperature", min_value=0.0, max_value=2.0)
        assert section.fields["temperature"].field_type == FieldType.FLOAT

    def test_boolean_helper(self):
        """Test boolean() helper method."""
        section = ConfigSectionSchema()
        section.boolean("debug", False, "Debug mode")
        assert section.fields["debug"].field_type == FieldType.BOOLEAN

    def test_list_helper(self):
        """Test list() helper method."""
        section = ConfigSectionSchema()
        section.list("hosts", ["localhost"], "Allowed hosts")
        assert section.fields["hosts"].field_type == FieldType.LIST

    def test_path_helper(self):
        """Test path() helper method."""
        section = ConfigSectionSchema()
        section.path("models", "models", "Models directory")
        assert section.fields["models"].field_type == FieldType.PATH

    def test_method_chaining(self):
        """Test that helper methods can be chained."""
        section = ConfigSectionSchema()
        section.string("name", "test").integer("port", 8080).boolean("debug", False)
        assert len(section.fields) == 3

    def test_validate_section_defaults(self):
        """Test section validation applies defaults."""
        section = ConfigSectionSchema()
        section.string("name", "default_name")
        section.integer("port", 8080)

        result = section.validate_section({})
        assert result["name"] == "default_name"
        assert result["port"] == 8080

    def test_validate_section_with_values(self):
        """Test section validation with provided values."""
        section = ConfigSectionSchema()
        section.string("name", "default_name")
        section.integer("port", 8080)

        result = section.validate_section({"name": "custom", "port": "9000"})
        assert result["name"] == "custom"
        assert result["port"] == 9000

    def test_validate_section_unknown_fields(self):
        """Test section validation preserves unknown fields."""
        section = ConfigSectionSchema()
        section.string("name", "default")

        result = section.validate_section({"name": "test", "unknown": "value"})
        assert result["name"] == "test"
        assert result["unknown"] == "value"

    def test_validate_section_type_conversion(self):
        """Test section validation converts types."""
        section = ConfigSectionSchema()
        section.integer("port", 8080)
        section.boolean("debug", False)
        section.float("temperature", 0.7)

        result = section.validate_section({
            "port": "9000",
            "debug": "true",
            "temperature": "0.5"
        })
        assert result["port"] == 9000
        assert result["debug"] is True
        assert result["temperature"] == 0.5


class TestConfigSchema:
    """Tests for ConfigSchema class."""

    def test_schema_creation(self):
        """Test schema creation."""
        schema = ConfigSchema()
        assert schema.sections == {}

    def test_add_section(self):
        """Test adding a section."""
        schema = ConfigSchema()
        section = schema.add_section("server", description="Server settings")
        assert "server" in schema.sections
        assert isinstance(section, ConfigSectionSchema)

    def test_add_section_with_existing(self):
        """Test adding a pre-configured section."""
        schema = ConfigSchema()
        section = ConfigSectionSchema("Pre-configured")
        section.string("name", "test")

        result = schema.add_section("custom", section)
        assert result is section
        assert "custom" in schema.sections
        assert "name" in schema.sections["custom"].fields

    def test_get_section(self):
        """Test getting a section."""
        schema = ConfigSchema()
        schema.add_section("server")

        assert schema.get_section("server") is not None
        assert schema.get_section("nonexistent") is None

    def test_validate_applies_defaults(self):
        """Test schema validation applies defaults."""
        schema = ConfigSchema()
        server = schema.add_section("server")
        server.integer("port", 8080)
        server.boolean("debug", False)

        result = schema.validate({})
        assert result["server"]["port"] == 8080
        assert result["server"]["debug"] is False

    def test_validate_with_values(self):
        """Test schema validation with provided values."""
        schema = ConfigSchema()
        server = schema.add_section("server")
        server.integer("port", 8080)

        result = schema.validate({"server": {"port": "9000"}})
        assert result["server"]["port"] == 9000

    def test_validate_unknown_sections(self):
        """Test schema validation preserves unknown sections."""
        schema = ConfigSchema()
        schema.add_section("server")

        result = schema.validate({
            "server": {},
            "custom": {"key": "value"}
        })
        assert "custom" in result
        assert result["custom"]["key"] == "value"


class TestRKLLAMASchema:
    """Tests for the RKLLAMA-specific schema."""

    def test_schema_creation(self):
        """Test RKLLAMA schema is created correctly."""
        schema = create_rkllama_schema()
        assert isinstance(schema, ConfigSchema)

    def test_server_section_exists(self):
        """Test server section exists with expected fields."""
        schema = create_rkllama_schema()
        server = schema.get_section("server")

        assert server is not None
        assert "port" in server.fields
        assert "host" in server.fields
        assert "debug" in server.fields

    def test_server_port_validation(self):
        """Test server port has correct constraints."""
        schema = create_rkllama_schema()
        port_field = schema.get_section("server").fields["port"]

        assert port_field.default == 8080
        assert port_field.min_value == 1
        assert port_field.max_value == 65535

    def test_paths_section_exists(self):
        """Test paths section exists with expected fields."""
        schema = create_rkllama_schema()
        paths = schema.get_section("paths")

        assert paths is not None
        assert "models" in paths.fields
        assert "logs" in paths.fields
        assert "data" in paths.fields
        assert "lib" in paths.fields
        assert "temp" in paths.fields

    def test_model_section_exists(self):
        """Test model section exists with expected fields."""
        schema = create_rkllama_schema()
        model = schema.get_section("model")

        assert model is not None
        assert "default_temperature" in model.fields
        assert "default_num_ctx" in model.fields
        assert "default_max_new_tokens" in model.fields
        assert "default_top_k" in model.fields
        assert "default_top_p" in model.fields

    def test_platform_section_exists(self):
        """Test platform section exists with expected fields."""
        schema = create_rkllama_schema()
        platform = schema.get_section("platform")

        assert platform is not None
        assert "processor" in platform.fields

    def test_processor_options(self):
        """Test processor field has correct options."""
        schema = create_rkllama_schema()
        processor_field = schema.get_section("platform").fields["processor"]

        assert processor_field.default == "rk3588"
        assert processor_field.options == ["rk3588", "rk3576"]

    def test_processor_validation(self):
        """Test processor field validates correctly."""
        schema = create_rkllama_schema()
        processor_field = schema.get_section("platform").fields["processor"]

        assert processor_field.validate("rk3588") == "rk3588"
        assert processor_field.validate("rk3576") == "rk3576"

        with pytest.raises(ValueError):
            processor_field.validate("invalid_processor")

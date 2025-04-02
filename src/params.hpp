#pragma once

#include "misc/common.hpp"

namespace img_aligner::params
{

    // every Param contains a Value which is a std::variant of different classes
    // defined below.

    // example values for common constructor arguments
    // id: "target_temperature"
    // name: "Target Temperature"
    // cli_option: "-t,--temp", empty string will omit param from the CLI
    // cli_behavior: "None" will omit param from the CLI
    // gui_behavior: "None" will omit param from the GUI

    class Param;

    // if and how a bool value should be represented in the CLI
    enum class BoolCliBehavior
    {
        None,
        Flag, // initial value must be false because flags can only turn on
        Toggle // enabling the flag inverts the value
    };

    // if a bool value should be represented in the GUI
    enum class BoolGuiBehavior
    {
        None,
        Checkbox
    };

    // if a numeric value should be represented in the CLI
    enum class NumericValueCliBehavior
    {
        None,
        Option
    };

    // if and how a numeric value should be represented in the GUI
    enum class NumericValueGuiBehavior
    {
        None,
        Input,
        Drag,
        Slider
    };

    // if and how a bool value should be represented in the CLI
    enum class TextCliBehavior
    {
        None,
        Option
    };

    // if a bool value should be represented in the GUI
    enum class TextGuiBehavior
    {
        None,
        Input
    };

    // where a callback was called from
    enum class InvokeSource
    {
        Cli,
        Gui,
        MetadataExport
    };

    template<
        typename T,
        typename CliBehaviorEnum,
        typename GuiBehaviorEnum,
        bool can_have_min_max
    >
    class NumericValue
    {
    private:
        static constexpr auto S_ERROR_MIN_MAX_NOT_ALLOWED =
            "min/max values aren't allowed for this numeric value type";

    public:
        NumericValue(
            std::string cli_option,
            const CliBehaviorEnum& cli_behavior,
            const GuiBehaviorEnum& gui_behavior,
            T initial_val,
            std::optional<T> min = std::nullopt,
            std::optional<T> max = std::nullopt
        )
            : _cli_option(std::move(cli_option)),
            _cli_behavior(cli_behavior),
            _gui_behavior(gui_behavior),
            _val(std::move(initial_val)),
            _min(std::move(min)),
            _max(std::move(max))
        {
            if (!can_have_min_max && (min || max))
            {
                throw std::invalid_argument(S_ERROR_MIN_MAX_NOT_ALLOWED);
            }
        }

        constexpr const std::string& cli_option() const
        {
            return _cli_option;
        }

        constexpr const CliBehaviorEnum& cli_behavior() const
        {
            return _cli_behavior;
        };

        constexpr const GuiBehaviorEnum& gui_behavior() const
        {
            return _gui_behavior;
        };

        constexpr const T& get() const
        {
            return _val;
        }

        void set(T new_val)
        {
            _val = std::move(new_val);
            if constexpr (can_have_min_max)
            {
                if (min())
                {
                    _val = std::max(_val, *min());
                }
                if (max())
                {
                    _val = std::min(_val, *max());
                }
            }
        }

        constexpr const std::optional<T>& min() const
        {
            if constexpr (!can_have_min_max)
            {
                throw std::invalid_argument(S_ERROR_MIN_MAX_NOT_ALLOWED);
            }
            return _min;
        }

        void set_min(std::optional<T> new_min)
        {
            if constexpr (!can_have_min_max)
            {
                throw std::invalid_argument(S_ERROR_MIN_MAX_NOT_ALLOWED);
            }
            _min = std::move(new_min);
        }

        constexpr const std::optional<T>& max() const
        {
            if constexpr (!can_have_min_max)
            {
                throw std::invalid_argument(S_ERROR_MIN_MAX_NOT_ALLOWED);
            }
            return _max;
        }

        void set_max(std::optional<T> new_max)
        {
            if constexpr (!can_have_min_max)
            {
                throw std::invalid_argument(S_ERROR_MIN_MAX_NOT_ALLOWED);
            }
            _max = std::move(new_max);
        }

    private:
        std::string _cli_option;
        CliBehaviorEnum _cli_behavior;
        GuiBehaviorEnum _gui_behavior;

        T _val;
        std::optional<T> _min;
        std::optional<T> _max;

    };

    using Bool = NumericValue<
        bool,
        BoolCliBehavior,
        BoolGuiBehavior,
        false
    >;
    using I32 = NumericValue<
        int32_t,
        NumericValueCliBehavior,
        NumericValueGuiBehavior,
        true
    >;
    using U32 = NumericValue<
        uint32_t,
        NumericValueCliBehavior,
        NumericValueGuiBehavior,
        true
    >;
    using I64 = NumericValue<
        int64_t,
        NumericValueCliBehavior,
        NumericValueGuiBehavior,
        true
    >;
    using U64 = NumericValue<
        uint64_t,
        NumericValueCliBehavior,
        NumericValueGuiBehavior,
        true
    >;
    using F32 = NumericValue<
        float,
        NumericValueCliBehavior,
        NumericValueGuiBehavior,
        true
    >;
    using F64 = NumericValue<
        double,
        NumericValueCliBehavior,
        NumericValueGuiBehavior,
        true
    >;

    class Text
    {
    public:
        constexpr Text(
            std::string cli_option,
            const TextCliBehavior& cli_behavior,
            const TextGuiBehavior& gui_behavior,
            std::string initial_val
        )
            : _cli_option(std::move(cli_option)),
            _cli_behavior(cli_behavior),
            _gui_behavior(gui_behavior),
            _val(std::move(initial_val))
        {}

        constexpr const std::string& cli_option() const
        {
            return _cli_option;
        }

        constexpr const TextCliBehavior& cli_behavior() const
        {
            return _cli_behavior;
        };

        constexpr const TextGuiBehavior& gui_behavior() const
        {
            return _gui_behavior;
        };

        constexpr const std::string& get() const
        {
            return _val;
        }

        constexpr void set(std::string_view new_val)
        {
            _val = std::move(_val);
        }

    private:
        std::string _cli_option;
        TextCliBehavior _cli_behavior;
        TextGuiBehavior _gui_behavior;

        std::string _val;

    };

    class Label
    {
    public:
        constexpr Label(std::string text, bool bold)
            : _text(std::move(text)),
            _bold(bold)
        {}

        constexpr const std::string& text() const
        {
            return _text;
        }

        constexpr void set_text(std::string_view new_text)
        {
            _text = new_text;
        }

        constexpr bool bold() const
        {
            return _bold;
        }

        constexpr void set_bold(bool new_bold)
        {
            _bold = new_bold;
        }

    private:
        std::string _text;
        bool _bold;

    };

    // this is for labels whose text might change. the callback should return
    // the current text.
    class DynamicLabel
    {
    public:
        inline DynamicLabel(
            std::function<std::string(const Param&, InvokeSource)> callback,
            bool bold
        )
            : _callback(std::move(callback)),
            _bold(bold)
        {}

        constexpr const std::function<std::string(const Param&, InvokeSource)>&
            callback() const
        {
            return _callback;
        }

        inline void set_callback(
            std::function<std::string(const Param&, InvokeSource)> new_callback
        )
        {
            _callback = std::move(new_callback);
        }

        inline std::string text(
            const Param& source_param,
            InvokeSource invoke_source
        ) const
        {
            return callback()(source_param, invoke_source);
        }

        constexpr bool bold() const
        {
            return _bold;
        }

        constexpr void set_bold(bool new_bold)
        {
            _bold = new_bold;
        }

    private:
        std::function<std::string(const Param&, InvokeSource)> _callback;
        bool _bold;

    };

    // divider
    class Div
    {
    public:
        constexpr Div(bool small)
            : _small(small)
        {}

        constexpr bool small() const
        {
            return _small;
        }

        constexpr void set_small(bool new_small)
        {
            _small = new_small;
        }

    private:
        bool _small;

    };

    class Button
    {
    public:
        inline Button(
            std::function<void(const Param&, InvokeSource)> callback
        )
            : _callback(std::move(callback))
        {}

        constexpr const std::function<void(const Param&, InvokeSource)>&
            callback() const
        {
            return _callback;
        }

        inline void set_callback(
            std::function<void(const Param&, InvokeSource)> new_callback
        )
        {
            _callback = std::move(new_callback);
        }

        inline void invoke(
            const Param& source_param,
            InvokeSource invoke_source
        ) const
        {
            callback()(source_param, invoke_source);
        }

    private:
        std::function<void(const Param&, InvokeSource)> _callback;

    };

    // this type of value simply runs some code when it's accessed. in the GUI,
    // the callback will be called every frame. in the CLI and when exporting
    // metadata, the callback will be called once.
    class Invokable
    {
    public:
        inline Invokable(
            std::function<void(const Param&, InvokeSource)> callback
        )
            : _callback(std::move(callback))
        {}

        constexpr const std::function<void(const Param&, InvokeSource)>&
            callback() const
        {
            return _callback;
        }

        inline void set_callback(
            std::function<void(const Param&, InvokeSource)> new_callback
        )
        {
            _callback = std::move(new_callback);
        }

        inline void invoke(
            const Param& source_param,
            InvokeSource invoke_source
        ) const
        {
            callback()(source_param, invoke_source);
        }

    private:
        std::function<void(const Param&, InvokeSource)> _callback;

    };

    using Value = std::variant<
        Bool,
        I32,
        U32,
        I64,
        U64,
        F32,
        F64,
        Text,
        Label,
        DynamicLabel,
        Div,
        Button,
        Invokable
    >;

    class Param
    {
    public:
        Param(
            std::string id,
            std::string name,
            Value initial_val
        )
            : _id(std::move(id)),
            _name(std::move(name)),
            _val(std::move(initial_val))
        {}

        constexpr const std::string& id() const
        {
            return _id;
        }

        constexpr const std::string& name() const
        {
            return _name;
        }

        constexpr void set_name(std::string new_name)
        {
            _name = std::move(new_name);
        }

        constexpr const Value& val() const
        {
            return _val;
        }

        constexpr Value& val()
        {
            return _val;
        }

    private:
        std::string _id;
        std::string _name;
        Value _val;

    };

    // this is simply a list of parameters
    class Section
    {
    public:
        constexpr Section(std::string id, std::string name)
            : _id(std::move(id)), _name(std::move(name))
        {}

        constexpr const std::string& id() const
        {
            return _id;
        }

        constexpr const std::string& name() const
        {
            return _name;
        }

        constexpr const Param& operator[](std::size_t idx) const
        {
            if (idx >= _params.size())
            {
                throw std::range_error("index out of bounds");
            }
            return _params[idx];
        }

        constexpr Param& operator[](std::size_t idx)
        {
            if (idx >= _params.size())
            {
                throw std::range_error("index out of bounds");
            }
            return _params[idx];
        }

        constexpr const Param& operator[](std::string_view id) const
        {
            for (const auto& param : _params)
            {
                if (param.id() == id)
                {
                    return param;
                }
            }
            throw std::range_error("parameter ID not found");
        }

        constexpr Param& operator[](std::string_view id)
        {
            for (auto& param : _params)
            {
                if (param.id() == id)
                {
                    return param;
                }
            }
            throw std::range_error("parameter ID not found");
        }

        constexpr size_t size() const
        {
            return _params.size();
        }

        constexpr bool contains(std::string_view id) const
        {
            for (const auto& param : _params)
            {
                if (param.id() == id)
                {
                    return true;
                }
            }
            return false;
        }

        inline Param& add(Param param)
        {
            if (contains(param.id()))
            {
                throw std::invalid_argument(std::format(
                    "can't add new parameter because a parameter with the same "
                    "ID ({}) already exists",
                    param.id()
                ).c_str());
            }
            _params.push_back(std::move(param));
            return _params.back();
        }

        inline Param& add(Param param, size_t idx)
        {
            if (contains(param.id()))
            {
                throw std::invalid_argument(std::format(
                    "can't add new parameter because a parameter with the same "
                    "ID ({}) already exists",
                    param.id()
                ).c_str());
            }
            if (idx > _params.size())
            {
                throw std::range_error("index out of bounds");
            }
            _params.insert(_params.begin() + idx, std::move(param));
            return _params[idx];
        }

        inline bool remove(const Param& param)
        {
            return remove(param.id());
        }

        inline bool remove(std::string_view id)
        {
            for (size_t i = 0; i < _params.size(); i++)
            {
                if (_params[i].id() == id)
                {
                    remove(i);
                    return true;
                }
            }
            return false;
        }

        inline void remove(size_t idx)
        {
            if (idx >= _params.size())
            {
                throw std::range_error("index out of bounds");
            }
            _params.erase(_params.begin() + idx);
        }

    private:
        std::string _id;
        std::string _name;

        std::vector<Param> _params;

    };

    // this is simply a list of sections
    class SectionList
    {
    public:
        SectionList() = default;

        constexpr const Section& operator[](std::size_t idx) const
        {
            if (idx >= _sections.size())
            {
                throw std::range_error("index out of bounds");
            }
            return _sections[idx];
        }

        constexpr Section& operator[](std::size_t idx)
        {
            if (idx >= _sections.size())
            {
                throw std::range_error("index out of bounds");
            }
            return _sections[idx];
        }

        constexpr const Section& operator[](std::string_view id) const
        {
            for (const auto& section : _sections)
            {
                if (section.id() == id)
                {
                    return section;
                }
            }
            throw std::range_error("section ID not found");
        }

        constexpr Section& operator[](std::string_view id)
        {
            for (auto& section : _sections)
            {
                if (section.id() == id)
                {
                    return section;
                }
            }
            throw std::range_error("section ID not found");
        }

        constexpr size_t size() const
        {
            return _sections.size();
        }

        constexpr bool contains(std::string_view id) const
        {
            for (const auto& section : _sections)
            {
                if (section.id() == id)
                {
                    return true;
                }
            }
            return false;
        }

        inline Section& add(Section section)
        {
            if (contains(section.id()))
            {
                throw std::invalid_argument(std::format(
                    "can't add new section because a section with the same ID "
                    "({}) already exists",
                    section.id()
                ).c_str());
            }
            _sections.push_back(std::move(section));
            return _sections.back();
        }

        inline Section& add(Section section, size_t idx)
        {
            if (contains(section.id()))
            {
                throw std::invalid_argument(std::format(
                    "can't add new section because a section with the same ID "
                    "({}) already exists",
                    section.id()
                ).c_str());
            }
            if (idx > _sections.size())
            {
                throw std::range_error("index out of bounds");
            }
            _sections.insert(_sections.begin() + idx, std::move(section));
            return _sections[idx];
        }

        inline bool remove(const Section& section)
        {
            return remove(section.id());
        }

        inline bool remove(std::string_view id)
        {
            for (size_t i = 0; i < _sections.size(); i++)
            {
                if (_sections[i].id() == id)
                {
                    remove(i);
                    return true;
                }
            }
            return false;
        }

        inline void remove(size_t idx)
        {
            if (idx >= _sections.size())
            {
                throw std::range_error("index out of bounds");
            }
            _sections.erase(_sections.begin() + idx);
        }

    private:
        std::vector<Section> _sections;

    };

}

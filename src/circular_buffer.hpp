#pragma once

#include <algorithm>
#include <stdexcept>
#include <cstdint>

namespace img_aligner
{

    template<typename T, size_t _capacity>
        requires (_capacity > 0)
    class CircularBuffer
    {
    public:
        CircularBuffer() = default;

        CircularBuffer(const T& default_value)
        {
            std::fill(_data, _data + _capacity, default_value);
        }

        constexpr size_t capacity() const
        {
            return _capacity;
        }

        constexpr size_t size() const
        {
            if (head >= tail)
            {
                return head - tail;
            }
            return _capacity - (tail - head);
        }

        constexpr bool empty() const
        {
            return head == tail;
        }

        constexpr bool full() const
        {
            return ((head + 1) % _capacity) == tail;
        }

        constexpr T& front()
        {
            if (empty())
            {
                throw std::out_of_range(
                    "there's no front value if circular buffer is empty"
                );
            }
            return _data[tail];
        }

        constexpr const T& front() const
        {
            if (empty())
            {
                throw std::out_of_range(
                    "there's no front value if circular buffer is empty"
                );
            }
            return _data[tail];
        }

        constexpr T& back()
        {
            if (empty())
            {
                throw std::out_of_range(
                    "there's no back value if circular buffer is empty"
                );
            }
            return _data[((ptrdiff_t)head - 1) % (ptrdiff_t)_capacity];
        }

        constexpr const T& back() const
        {
            if (empty())
            {
                throw std::out_of_range(
                    "there's no back value if circular buffer is empty"
                );
            }
            return _data[((ptrdiff_t)head - 1) % (ptrdiff_t)_capacity];
        }

        constexpr void push_back(const T& v)
        {
            _data[head] = v;

            head = (head + 1) % _capacity;
            if (head == tail)
            {
                tail = (tail + 1) % _capacity;
            }
        }

        // you are advised to make a copy of the returned reference because it
        // can get rewritten.
        constexpr T& pop_front()
        {
            if (empty())
            {
                throw std::out_of_range(
                    "can't pop value from circular buffer if it's empty"
                );
            }
            T& v = _data[tail];
            tail = (tail + 1) % _capacity;
            return v;
        }

    private:
        T _data[_capacity];
        size_t head = 0;
        size_t tail = 0;

    };

}

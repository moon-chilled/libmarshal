
// Copyright 2007 Edd Dawson.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

"",
"//! This class is used as an exception. file_not_found objects are thrown whenever an attempt is",
"//! made to open a non-existent embedded file.",
"class file_not_found : public std::runtime_error",
"{",
"    public:",
"        //! Create a file_not_found object whose message indicates that the given file could",
"        //! not be found embedded within the application",
"        file_not_found(const std::string &filename);",
"",
"        //! Destroys the object mercilessly",
"        ~file_not_found() throw();",
"};",
"",
"//! file objects represent the files embedded in the application. You can access the data of a",
"//! file via either a standard C++ stream interface of through a direct pointer-to-char.",
"//!",
"//! You can have as many files as needed accessing the same data in a program. They won't",
"//! interfere with one another.",
"class file",
"{",
"    public:",
"        //! Used to iterate over the bytes of the file as const chars. You should not rely on this",
"        //! being a pointer for compatibility with future versions",
"        typedef const char *iterator;",
"",
"        //! Used to iterate over the bytes of the file as const chars. You should not rely on this",
"        //! being a pointer for compatibility with future versions",
"        typedef const char *const_iterator;",
"",
"        //! size() returns objects of this unsigned integral type",
"        typedef std::size_t size_type;",
"",
"        //! Attempts to construct a file to access the data with the given name. If no such file",
"        //! has been embedded, a file_not_found exception is thrown.",
"        file(const std::string &filename); // throw(file_not_found)",
"",
"        //! Copies a file. The copy will have its stream reset",
"        file(const file &other);",
"",
"        //! Assigns a copy of other to this file, with its stream reset",
"        file &operator= (const file &other);",
"",
"        //! Destroys the file",
"        ~file();",
"",
"        //! Swaps the data that this file and other refer to",
"        void swap(file &other);",
"",
"        //! Returns the name of the file, as passed to the constructor",
"        const char *name() const;",
"",
"        //! Returns a stream that may be used to access the contents of the file",
"        std::istream &istream();",
"",
"        //! Resets the state of the stream so that data can be read again",
"        void reset_stream();",
"",
"        //! Returns an iterator that refers to the first byte in the data stream of the file",
"        const_iterator begin() const;",
"",
"        //! Returns an iterator that refers one-past-the-last byte in the data stream of the",
"        //! file",
"        const_iterator end() const;",
"",
"        //! Returns the number of bytes in the file",
"        size_type size() const;",
"",
"    private:",
"        struct impl;",
"        impl *pimpl_;",
"};",
"",
"//! file_name_iterators can be used to enumerate the names of all the files embedded within an",
"//! application. Dereferencing a file_name_iterator yields a const char *, naming a file. The names",
"//! are not organised in any particular order.",
"//!",
"//! This is a bidirectional iterator.",
"class file_name_iterator :",
"    public std::iterator<std::bidirectional_iterator_tag, const char * const>",
"{",
"    public:",
"        //! Pre-increment, with the usual semantics",
"        file_name_iterator &operator++ ();",
"",
"        //! Post-increment, with the usual semantics",
"        file_name_iterator operator++ (int);",
"",
"        //! Pre-decrement, with the usual semantics",
"        file_name_iterator &operator-- ();",
"",
"        //! Post-decrement, with the usual semantics",
"        file_name_iterator operator-- (int);",
"",
"        //! Dereferencing operator. Returns the name that this iterator refers to",
"        reference operator* () const;",
"",
"        //! Dereferencing operator. Returns a pointer to the name that this iterator refers to",
"        pointer operator-> () const;",
"",
"        //! Returns true if and only if the two iterators refer to the same file name",
"        bool operator== (const file_name_iterator &rhs) const;",
"",
"        //! Returns false if and only if the two iterators refer to the same file name",
"        bool operator!= (const file_name_iterator &rhs) const;",
"",
"        //! Returns an iterator that refers to the first name in the internal sequence of file names,",
"        //! or end() of no files have been embedded.",
"        static file_name_iterator begin();",
"",
"        //! Returns an iterator that refers to one-past-the-final element in the sequence of names.",
"        //! This iterator or any iterator equal to it should not be dereferenced.",
"        static file_name_iterator end();",
"",
"    private:",
"        file_name_iterator(const void *p);",
"        const void *p_;",
"};",
"",
""

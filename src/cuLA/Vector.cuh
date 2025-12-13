#ifndef CULA_VECTOR_CUH
#define CULA_VECTOR_CUH

struct Vector {
	int count;
	float* device_data;

	Vector(int count);

	~Vector();

	inline size_t size() const {
		return count;
	}

	inline size_t size_bytes() const {
		return size() * sizeof(float);
	}

	void upload(const float* h_data);

	void download(float* h_data) const;
};

#endif // CULA_VECTOR_CUH
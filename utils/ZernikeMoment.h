#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;

namespace MtZernikeMoment
{
	class ZernikeMomentFailure
	{
	public:
		ZernikeMomentFailure(const char* msg);
		const char* GetMessage();
	private:
		const char* Message;
	};

	class ZernikeMoment
	{
	public:
	
		ZernikeMoment(const cv::Mat& image) :mImage(image) {}
		void compute(std::vector<float>& x, std::vector<float>& y)throw(ZernikeMomentFailure);
		
	private:
		cv::Mat mImage;
		cv::Mat mGrayImage;

		int GN = 7;
		double pi = 3.14159265358979323846;

		cv::Mat M00 = (cv::Mat_<double>(7, 7) <<
			0, 0.0287, 0.0686, 0.0807, 0.0686, 0.0287, 0,
			0.0287, 0.0815, 0.0816, 0.0816, 0.0816, 0.0815, 0.0287,
			0.0686, 0.0816, 0.0816, 0.0816, 0.0816, 0.0816, 0.0686,
			0.0807, 0.0816, 0.0816, 0.0816, 0.0816, 0.0816, 0.0807,
			0.0686, 0.0816, 0.0816, 0.0816, 0.0816, 0.0816, 0.0686,
			0.0287, 0.0815, 0.0816, 0.0816, 0.0816, 0.0815, 0.0287,
			0, 0.0287, 0.0686, 0.0807, 0.0686, 0.0287, 0);

		cv::Mat M11R = (cv::Mat_<double>(7, 7) <<
			0, -0.015, -0.019, 0, 0.019, 0.015, 0,
			-0.0224, -0.0466, -0.0233, 0, 0.0233, 0.0466, 0.0224,
			-0.0573, -0.0466, -0.0233, 0, 0.0233, 0.0466, 0.0573,
			-0.069, -0.0466, -0.0233, 0, 0.0233, 0.0466, 0.069,
			-0.0573, -0.0466, -0.0233, 0, 0.0233, 0.0466, 0.0573,
			-0.0224, -0.0466, -0.0233, 0, 0.0233, 0.0466, 0.0224,
			0, -0.015, -0.019, 0, 0.019, 0.015, 0);

		cv::Mat M11I = (cv::Mat_<double>(7, 7) <<
			0, -0.0224, -0.0573, -0.069, -0.0573, -0.0224, 0,
			-0.015, -0.0466, -0.0466, -0.0466, -0.0466, -0.0466, -0.015,
			-0.019, -0.0233, -0.0233, -0.0233, -0.0233, -0.0233, -0.019,
			0, 0, 0, 0, 0, 0, 0,
			0.019, 0.0233, 0.0233, 0.0233, 0.0233, 0.0233, 0.019,
			0.015, 0.0466, 0.0466, 0.0466, 0.0466, 0.0466, 0.015,
			0, 0.0224, 0.0573, 0.069, 0.0573, 0.0224, 0);

		cv::Mat M20 = (cv::Mat_<double>(7, 7) <<
			0, 0.0225, 0.0394, 0.0396, 0.0394, 0.0225, 0,
			0.0225, 0.0271, -0.0128, -0.0261, -0.0128, 0.0271, 0.0225,
			0.0394, -0.0128, -0.0528, -0.0661, -0.0528, -0.0128, 0.0394,
			0.0396, -0.0261, -0.0661, -0.0794, -0.0661, -0.0261, 0.0396,
			0.0394, -0.0128, -0.0528, -0.0661, -0.0528, -0.0128, 0.0394,
			0.0225, 0.0271, -0.0128, -0.0261, -0.0128, 0.0271, 0.0225,
			0, 0.0225, 0.0394, 0.0396, 0.0394, 0.0225, 0);

		cv::Mat M31R = (cv::Mat_<double>(7, 7) <<
			0, -0.0103, -0.0073, 0, 0.0073, 0.0103, 0,
			-0.0153, -0.0018, 0.0162, 0, -0.0162, 0.0018, 0.0153,
			-0.0223, 0.0324, 0.0333, 0, -0.0333, -0.0324, 0.0223,
			-0.0190, 0.0438, 0.0390, 0, -0.0390, -0.0438, 0.0190,
			-0.0223, 0.0324, 0.0333, 0, -0.0333, -0.0324, 0.0223,
			-0.0153, -0.0018, 0.0162, 0, -0.0162, 0.0018, 0.0153,
			0, -0.0103, -0.0073, 0, 0.0073, 0.0103, 0);

		cv::Mat M31I = (cv::Mat_<double>(7, 7) <<
			0, -0.0153, -0.0223, -0.019, -0.0223, -0.0153, 0,
			-0.0103, -0.0018, 0.0324, 0.0438, 0.0324, -0.0018, -0.0103,
			-0.0073, 0.0162, 0.0333, 0.039, 0.0333, 0.0162, -0.0073,
			0, 0, 0, 0, 0, 0, 0,
			0.0073, -0.0162, -0.0333, -0.039, -0.0333, -0.0162, 0.0073,
			0.0103, 0.0018, -0.0324, -0.0438, -0.0324, 0.0018, 0.0103,
			0, 0.0153, 0.0223, 0.0190, 0.0223, 0.0153, 0);

		cv::Mat M40 = (cv::Mat_<double>(7, 7) <<
			0, 0.013, 0.0056, -0.0018, 0.0056, 0.013, 0,
			0.0130, -0.0186, -0.0323, -0.0239, -0.0323, -0.0186, 0.0130,
			0.0056, -0.0323, 0.0125, 0.0406, 0.0125, -0.0323, 0.0056,
			-0.0018, -0.0239, 0.0406, 0.0751, 0.0406, -0.0239, -0.0018,
			0.0056, -0.0323, 0.0125, 0.0406, 0.0125, -0.0323, 0.0056,
			0.0130, -0.0186, -0.0323, -0.0239, -0.0323, -0.0186, 0.0130,
			0, 0.013, 0.0056, -0.0018, 0.0056, 0.013, 0);

		void MedianBlur(const cv::Mat& source, cv::Mat& target, const int& windows);
	};

	class MaxHeap
	{
	public:

		MaxHeap()
		{
			heap.resize(2);
			heapSize = 0;
		}

		void push(const int& x)
		{
			if (heapSize == 0) heap.resize(2);
			if (heapSize == heap.size() - 1) changeLength();

			int currentNode = ++heapSize;
			while (currentNode != 1 && x > heap[currentNode / 2])
			{
				heap[currentNode] = heap[currentNode / 2];
				currentNode /= 2;
			}
			heap[currentNode] = x;
		}

		void pop()
		{
			int deleteIndex = heapSize;
			int lastElement = heap[heapSize--];

			int currentNode = 1;
			int chirld = 2;
			while (chirld <= heapSize)
			{
				if (chirld < heapSize && heap[chirld] < heap[chirld + 1]) chirld++;
				if (lastElement >= heap[chirld]) break;

				heap[currentNode] = heap[chirld];
				currentNode = chirld;
				chirld *= 2;
			}
			heap[currentNode] = lastElement;
			heap.erase(heap.begin() + deleteIndex);
		}

		int size()
		{
			return heapSize;
		}

		int top()
		{
			if (!this->empty()) return heap[1];
			else exit(-1);
		}

		bool empty()
		{
			if (heapSize == 0) return true;
			return false;
		}

	private:
		std::vector<int> heap;
		int heapSize;

		void changeLength()
		{
			std::vector<int> heapTemp;
			heapTemp = heap;
			heap.resize(2 * heap.size());

			for (int i = 0; i < heapTemp.size(); i++)
			{
				heap[i] = heapTemp[i];
			}
		}
	};

	class MinHeap
	{
	public:

		MinHeap()
		{
			heap.resize(2);
			heapSize = 0;
		}

		void push(const int& x)
		{
			if (heapSize == 0) heap.resize(2);
			if (heapSize == heap.size() - 1) changeLength();

			int currentNode = ++heapSize;
			while (currentNode != 1 && x < heap[currentNode / 2])
			{
				heap[currentNode] = heap[currentNode / 2];
				currentNode /= 2;
			}
			heap[currentNode] = x;
		}

		void pop()
		{
			int deleteIndex = heapSize;
			int lastElement = heap[heapSize--];

			int currentNode = 1;
			int chirld = 2;
			while (chirld <= heapSize)
			{
				if (chirld < heapSize && heap[chirld] > heap[chirld + 1]) chirld++;
				if (lastElement <= heap[chirld]) break;

				heap[currentNode] = heap[chirld];
				currentNode = chirld;
				chirld *= 2;
			}
			heap[currentNode] = lastElement;
			heap.erase(heap.begin() + deleteIndex);
		}

		int size()
		{
			return heapSize;
		}

		int top()
		{
			if (!this->empty()) return heap[1];
			else exit(-1);
		}

		bool empty()
		{
			if (heapSize == 0) return true;
			return false;
		}

	private:
		std::vector<int> heap;
		int heapSize;

		void changeLength()
		{
			std::vector<int> heapTemp;
			heapTemp = heap;
			heap.resize(2 * heap.size());

			for (int i = 0; i < heapTemp.size(); i++)
			{
				heap[i] = heapTemp[i];
			}
		}
	};
}
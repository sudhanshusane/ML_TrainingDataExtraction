#include "cnpy.h"
#include <complex>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkDataArray.h>
#include <vtkSmartPointer.h>
#include <vtkRectilinearGridReader.h>
#include <vtkRectilinearGrid.h>
#include <vtkStructuredGrid.h>
#include <vtkGradientFilter.h>
#include <vtkDataSet.h>
#include <vtkDataSetWriter.h>
#include <vtkCellDataToPointData.h>
#include <sstream>
#include <iostream>
#include <vector>
#include <omp.h>
#include <vtkm/Matrix.h>
#include <vtkm/Types.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/io/VTKDataSetWriter.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/cont/DynamicCellSet.h>
#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleIndex.h>
#include <vtkm/cont/ArrayPortalToIterators.h>
#include <vtkm/cont/DeviceAdapter.h>
#include <vtkm/cont/ErrorFilterExecution.h>
#include <vtkm/worklet/ParticleAdvection.h>
#include <vtkm/worklet/particleadvection/GridEvaluators.h>
#include <vtkm/worklet/particleadvection/Integrators.h>
#include <vtkm/worklet/particleadvection/Particles.h>
#include <vtkm/cont/VariantArrayHandle.h>
#include <vtkm/cont/ArrayHandleVirtual.h>
#include <vtkm/worklet/lcs/GridMetaData.h>
#include <vtkm/worklet/lcs/LagrangianStructureHelpers.h>
#include <vtkm/worklet/LagrangianStructures.h>

#define PI 3.14159265
#define DIM_THETA 60
#define DIM_PHI 1

float calculateEntropy(int *bins, int num_bins)
{ 
  float probability[num_bins];
  int total_sum = 0; 
  for(int i = 0; i < num_bins; i++)
  { 
    total_sum += bins[i];
  }

  for(int i = 0; i < num_bins; i++)
  { 
    probability[i] = bins[i]/(total_sum*1.0);
  }
  
  float entropy = 0.0;
  
  for(int i = 0; i < num_bins; i++)
  { 
    if(probability[i] > 0.0)
      entropy -= (probability[i]*log2(probability[i]));
  }
  
  return entropy;
}

void estimateDistribution(int *bins, int num_bins, double* x, double* y, double* z, int N)
{
  /* There are DIM_THETA * DIM_PHI bins */
  
  double theta_range = 360.0/DIM_THETA; // 12
  double phi_range = 180.0/DIM_PHI; // 12

  /* BINS are ordered by increasing PHI
 *   // 0 [0,0] to [theta_range + 0 , phi_range + 0] .. 1 [0, phi_range + 0] to [theta_range + 0, phi_range*2 + 0]
 *
 *     // Theta_index = theta/theta_range; */

  int num_samples = N*N;
  for(int i = 0; i < num_samples; i++)
  {
    /* All values of theta and phi are between:
 *     // 0 < Theta < 360
 *         // 0 < Phi < 180 */

    double radius, theta, phi;
    radius = sqrt((x[i]*x[i]) + (y[i]*y[i]) + (z[i]*z[i]));
    if(radius > 0)
    {
      theta = (atan2(y[i],x[i]))*(180/PI);
      phi = (acos(z[i]/radius))*(180/PI);
      if(theta < 0)
        theta = 360 + theta; /* theta value is negative, i.e., -90 is the same as 270 */
    }
    else
    {
      theta = 0;
      phi = 0;
    }

    int t_index = theta/theta_range;
    int p_index = phi/phi_range;
    int bin_index = (t_index * DIM_PHI) + p_index;
    if(bin_index > DIM_THETA*DIM_PHI)
    {
      cout << "Indexing error" << endl;
    }
    else
    {
      bins[bin_index]++;
    }
/*    cout << "For : " << x[i] << "," << y[i] << "," << z[i] << " the value of radius is : " << radius << " the value of theta is : " << theta << " the value of phi is : " << phi << " the bin index is : " << bin_index << endl;*/
  }

}


void sampleNeighborhood(double *sample_x, double *sample_y, double *sample_z, int N, int *dims, int i, int j, int k, double *vec_x, double *vec_y, double *vec_z)
{
  int offset = (N-1)/2;
  int num_samples = N*N;
  int cnt = 0;

  if(dims[0] > N && dims[1] > N) // && dims[2] > N
  {
      /*    for(int r = k - offset; r < k + offset; r++)
 *          //    {    */
    for(int q = j - offset; q < j + offset; q++)
    {
      for(int p = i - offset; p < i + offset; p++) // x grows fastest
      {
            /* Left out
 *             // Right out
 *                         // Top out
 *                                     // Bottom out
 *                                                 // Front out
 *                                                             // Back out */
            int x_i, y_i, z_i;
            if(p < 0)
              x_i = -1*p;
            else if(p > (dims[0] -1))        
              x_i = (dims[0] - 1) - (p - (dims[0] - 1));
            else
              x_i = p;
            if(q < 0)
              y_i = -1*q;
            else if(q > (dims[1] -1))        
              y_i = (dims[1] - 1) - (q - (dims[1] - 1));
            else
              y_i = q;
         /*   if(r < 0)
 *                 z_i = -1*r;
 *                             else if(r > (dims[2] -1))        
 *                                           z_i = (dims[2] - 1) - (r - (dims[2] - 1));
 *                                                       else
 *                                                                    z_i = r;
 *                                                                               */

              z_i = k;

              int index = z_i*dims[1]*dims[0] + y_i*dims[0] + x_i;
              sample_x[cnt] = vec_x[index];
              sample_y[cnt] = vec_y[index];
              sample_z[cnt] = vec_z[index];
              cnt++;
      /*    } */
        }
      }
  }
  else
  {
    /* Neighborhood too large */
  }
}


int main(int argc, char* argv[])
{
	using Vec3f = vtkm::Vec<vtkm::FloatDefault, 3>;	
	
	std::stringstream s;
	s << argv[1] << argv[2] << argv[3]; // argv[2] should support multiple files.
	std::cout << "Inputfile: " << s.str() << std::endl;

	std::string output_file(argv[4]);

  vtkSmartPointer<vtkRectilinearGridReader> reader =
    vtkSmartPointer<vtkRectilinearGridReader>::New();
  reader->SetFileName(s.str().c_str());
  reader->Update();

  vtkSmartPointer<vtkRectilinearGrid> mesh =
    vtkSmartPointer<vtkRectilinearGrid>::New();
  mesh = reader->GetOutput();
  int num_pts = mesh->GetNumberOfPoints();
	int dims[3];
	dims[0] = atoi(argv[5]);
	dims[1] = atoi(argv[6]);
	dims[2] = atoi(argv[7]);

	int sdims[2];
	sdims[0] = atoi(argv[8]);
	sdims[1] = atoi(argv[9]);
 
	int num_intervals = atoi(argv[10]); // Need to be able to have interval options per num_steps
	int num_steps = atoi(argv[11]);  // Need to be able to set multiple options
	vtkm::Float32 step_size = atof(argv[12]); 

	int num_pts_slice = dims[0]*dims[1];	
	int num_seeds_interval = sdims[0]*sdims[1];
	int num_cells_interval = (sdims[0]-1)*(sdims[1]-1);
//  std::cout << "The number of points: " << num_pts << std::endl;

	vtkm::FloatDefault advectionTime = num_steps * step_size;

  vtkAbstractArray* a1 = mesh->GetPointData()->GetArray("u");
  vtkAbstractArray* a2 = mesh->GetPointData()->GetArray("v");

  vtkFloatArray* att1 = vtkFloatArray::SafeDownCast(a1);
  vtkFloatArray* att2 = vtkFloatArray::SafeDownCast(a2);

  float *xc = (float*) mesh->GetXCoordinates()->GetVoidPointer(0);
  float *yc = (float*) mesh->GetYCoordinates()->GetVoidPointer(0);
  float *zc = (float*) mesh->GetZCoordinates()->GetVoidPointer(0);

	vtkSmartPointer<vtkFloatArray> sxCoords = vtkSmartPointer<vtkFloatArray>::New();
	vtkSmartPointer<vtkFloatArray> syCoords = vtkSmartPointer<vtkFloatArray>::New();
	vtkSmartPointer<vtkFloatArray> szCoords = vtkSmartPointer<vtkFloatArray>::New();
	szCoords->InsertNextValue(0.0);

	float x_spacing = (xc[dims[0]-1] - xc[0])/(dims[0]-1);
	float y_spacing = (yc[dims[1]-1] - yc[0])/(dims[1]-1);
	float z_spacing = (zc[dims[2]-1] - zc[0])/(dims[2]-1);	
	
	float sx_spacing = (xc[dims[0]-1] - xc[0])/(sdims[0]-1);
	float sy_spacing = (yc[dims[1]-1] - yc[0])/(sdims[1]-1);

	float x_offset = sx_spacing/2.0;
	float y_offset = sy_spacing/2.0;

	for(int i = 0; i < sdims[0]; i++)
	{
		sxCoords->InsertNextValue(xc[0] + (i*sx_spacing));
	}
	
	for(int i = 0; i < sdims[1]; i++)
	{
		syCoords->InsertNextValue(yc[0] + (i*sy_spacing));
	}


/* Create output file */

	vtkSmartPointer<vtkDataSetWriter> writer =
		vtkSmartPointer<vtkDataSetWriter>::New();


	float* uni_x = (float*)malloc(sizeof(float)*num_seeds_interval);   // Seed locations for flow map
	float* uni_y = (float*)malloc(sizeof(float)*num_seeds_interval); 
	
	float* uni_sx = (float*)malloc(sizeof(float)*num_cells_interval);   // Seed locations for test and ground truth seeds
	float* uni_sy = (float*)malloc(sizeof(float)*num_cells_interval); 

	vtkm::cont::ArrayHandle<vtkm::FloatDefault> endpt;
	vtkm::cont::ArrayHandle<vtkm::Particle> flowmap_set, gt_set, test_set, test_reset;
	std::vector<double> ftle_val(num_seeds_interval);
	std::vector<double> error_val(num_seeds_interval);
	
	endpt.Allocate(num_cells_interval);
	flowmap_set.Allocate(num_seeds_interval*num_intervals);
	gt_set.Allocate(num_cells_interval*num_intervals);
	test_set.Allocate(num_cells_interval);
	test_reset.Allocate(num_cells_interval);
  	
	vtkm::cont::ArrayHandle<vtkm::Vec2f> lcsInputPoints, lcsOutputPoints;
	lcsInputPoints.Allocate(num_seeds_interval); // Equal to the size of flowmap set. FTLE is computed one flow map at a time. 
	lcsOutputPoints.Allocate(num_seeds_interval);

	// Compute the set of starting locations along a uniform grid for the flow map.
	for(int j = 0; j < sdims[1]; j++)
	{
		for(int i = 0; i < sdims[0]; i++)
		{
			int index = j*sdims[0] + i;
			uni_x[index] = xc[0] + i*sx_spacing;
			uni_y[index] = yc[0] + j*sy_spacing;
			
			lcsInputPoints.WritePortal().Set(index, vtkm::Vec2f(static_cast<vtkm::FloatDefault>(uni_x[index]), 
																													static_cast<vtkm::FloatDefault>(uni_y[index])));
		}
	}

	// Compute the set of starting locations along a uniform grid for the ground truth and test particles. 
	// We set gt at same time as flow map (later depending on number of intervals). 
	// We can set test now since test particles don't need time --- they interpolate flow maps.
	for(int j = 0; j < sdims[1]-1; j++)
	{
		for(int i = 0; i < sdims[0]-1; i++)
		{
			int index = j*(sdims[0]-1) + i;
			uni_sx[index] = xc[0] + i*sx_spacing + x_offset;
			uni_sy[index] = yc[0] + j*sy_spacing + y_offset;
			
			test_reset.WritePortal().Set(index, vtkm::Particle(Vec3f(static_cast<vtkm::FloatDefault>(uni_sx[index]), 
																														 static_cast<vtkm::FloatDefault>(uni_sy[index]), 
																														 static_cast<vtkm::FloatDefault>(0.0)), index));
			
			test_set.WritePortal().Set(index, vtkm::Particle(Vec3f(static_cast<vtkm::FloatDefault>(uni_sx[index]), 
																														 static_cast<vtkm::FloatDefault>(uni_sy[index]), 
																														 static_cast<vtkm::FloatDefault>(0.0)), index));
		}
	}

	int index = 0;
	int sindex = 0;
	for(int n = 0; n < num_intervals; n++)   // TODO: This will need to change if intervals are computed multiple times for varying step sizes. 
	{
		float time = n*step_size; // overlapping flow maps. Start times are 0, 0.01, 0.02, 0.03, .. 
		//float time = n*advectionTime; // non-overlapping flow maps. Start times are 0, 1, 2, 3, .. for 100 steps.
		for(int i = 0; i < num_seeds_interval; i++)
		{
			flowmap_set.WritePortal().Set(index, vtkm::Particle(Vec3f(static_cast<vtkm::FloatDefault>(uni_x[i]), 
																																static_cast<vtkm::FloatDefault>(uni_y[i]), 
																																static_cast<vtkm::FloatDefault>(time)), index));
			index++;
		}
		for(int i = 0; i < num_cells_interval; i++)
		{
			gt_set.WritePortal().Set(sindex, vtkm::Particle(Vec3f(static_cast<vtkm::FloatDefault>(uni_sx[i]), 
																														static_cast<vtkm::FloatDefault>(uni_sy[i]), 
																														static_cast<vtkm::FloatDefault>(time)), sindex));
			sindex++;
		}
	}

	std::cout << "Total number of flow map trajectories : " << index << " and expected is : " << num_seeds_interval*num_intervals << std::endl;
	std::cout << "Total number of ground truth trajectories : " << sindex << " and expected is : " << num_cells_interval*num_intervals << std::endl;

	vtkm::Id3 datasetDims(dims[0], dims[1], dims[2]);
  Vec3f origin3d(static_cast<vtkm::FloatDefault>(xc[0]),
  	             static_cast<vtkm::FloatDefault>(yc[0]),
								 static_cast<vtkm::FloatDefault>(zc[0]));
  Vec3f spacing3d(static_cast<vtkm::FloatDefault>(x_spacing),
                  static_cast<vtkm::FloatDefault>(y_spacing),
                  static_cast<vtkm::FloatDefault>(z_spacing));
	vtkm::cont::DataSet dataset;
	vtkm::cont::DataSetBuilderUniform uniformDatasetBuilder3d;
	dataset = uniformDatasetBuilder3d.Create(datasetDims, origin3d, spacing3d); 

	vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> velocity_field;
	vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> flowmap_field;
	velocity_field.Allocate(num_pts);
	flowmap_field.Allocate(num_seeds_interval);

	for(int i = 0; i < num_pts; i++)
	{
		velocity_field.WritePortal().Set(i, vtkm::Vec<vtkm::FloatDefault, 3>(att1->GetTuple1(i), att2->GetTuple1(i), 1.0));
	}

	std::cout << "Created velocity field" << std::endl;

/* Create the particle advection worklet set up */
	
	using FieldHandle3d = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>>;	
	const vtkm::cont::DynamicCellSet& cells3d = dataset.GetCellSet();
	const vtkm::cont::CoordinateSystem& coords3d = dataset.GetCoordinateSystem();
	using GridEvalType3d = vtkm::worklet::particleadvection::GridEvaluator<FieldHandle3d>;
	
	using RK4Type = vtkm::worklet::particleadvection::RK4Integrator<GridEvalType3d>;
	GridEvalType3d eval_flowmap(coords3d, cells3d, velocity_field);
	RK4Type rk4(eval_flowmap, static_cast<vtkm::Float32>(step_size));

	vtkm::cont::DataSet flowmap;
	vtkm::Id2 fmGridDims(sdims[0], sdims[1]);	
	vtkm::Vec2f origin2d(static_cast<vtkm::FloatDefault>(xc[0]),
									 static_cast<vtkm::FloatDefault>(yc[0]));
	vtkm::Vec2f spacing2d(static_cast<vtkm::FloatDefault>(sx_spacing),
										static_cast<vtkm::FloatDefault>(sy_spacing));
	vtkm::cont::DataSetBuilderUniform uniformDatasetBuilder2d;
	flowmap = uniformDatasetBuilder2d.Create(fmGridDims, origin2d, spacing2d);

/* Set up bilinear Lagrangian-based interpolation using existing VTK-m infrastructure. Use Euler step with step size = 1 and flow maps as input. */


	const vtkm::cont::DynamicCellSet& cells2d = flowmap.GetCellSet();
	const vtkm::cont::CoordinateSystem& coords2d = flowmap.GetCoordinateSystem();
	
	using LagrangianType = vtkm::worklet::particleadvection::EulerIntegrator<GridEvalType3d>;

	vtkm::worklet::ParticleAdvection particleadvection;
	vtkm::worklet::ParticleAdvectionResult res_flowmap, res_gt, res_test;

/* Compute the flow maps trajectories */

	res_flowmap = particleadvection.Run(rk4, flowmap_set, num_steps);
	auto flowmap_updated = res_flowmap.Particles;
	
	std::cout << "Computed flow map" << std::endl;

	res_gt = particleadvection.Run(rk4, gt_set, num_steps);
	auto gt_updated = res_gt.Particles;

	std::cout << "Computed ground truth" << std::endl;
	
	for(int i = 0; i < num_intervals; i++)
	{
		std::cout << "Starting iteration: " << i << std::endl;

//		vtkSmartPointer<vtkFloatArray> displacement_field = vtkSmartPointer<vtkFloatArray>::New();
//		std::stringstream s_d;
//		s_d << "displacement_" << i;
//		displacement_field->SetName(s_d.str().c_str());	

		for(int n = 0; n < num_seeds_interval; n++)
		{
			index = (i*num_seeds_interval) + n;  // flow map particles are selected by indexing by interval i. 
			auto end = flowmap_updated.ReadPortal().Get(index).Pos;
      float disp[2];
      disp[0] = end[0] - uni_x[n];
      disp[1] = end[1] - uni_y[n];
      lcsOutputPoints.WritePortal().Set(n, vtkm::Vec2f(static_cast<vtkm::FloatDefault>(end[0]),
                                                            static_cast<vtkm::FloatDefault>(end[1])));
      flowmap_field.WritePortal().Set(n, vtkm::Vec<vtkm::FloatDefault, 3>(static_cast<vtkm::FloatDefault>(disp[0]),
                                                                          static_cast<vtkm::FloatDefault>(disp[1]),
                                                                          static_cast<vtkm::FloatDefault>(0.0)));
	//		displacement_field->InsertNextValue(sqrt(pow(disp[0],2.0) + pow(disp[1],2.0)));
		}	
	//	std::cout << "Processed flow map for interval " << i << " and created distance field. " << std::endl;

		GridEvalType3d eval_lagrangian(coords2d, cells2d, flowmap_field);
		LagrangianType lagrangian(eval_lagrangian, static_cast<vtkm::Float32>(1.0));

		res_test = particleadvection.Run(lagrangian, test_set, 1);	
		auto test_updated = res_test.Particles;

		vtkSmartPointer<vtkFloatArray> endpt_cell_field = vtkSmartPointer<vtkFloatArray>::New();
		std::stringstream s_e;
		s_e << "endpt_c_" << i;
		endpt_cell_field->SetName(s_e.str().c_str());	

		for(int n = 0; n < num_cells_interval; n++)
		{
			sindex = (i*num_cells_interval) + n; // indexed by interval i since gt particles are all advected once ** IMPORTANT **  
			auto g = gt_updated.ReadPortal().Get(sindex).Pos;
			auto t = test_updated.ReadPortal().Get(n).Pos;
			float error = sqrt(pow((g[0]-t[0]),2.0) + pow((g[1]-t[1]),2.0));
			endpt_cell_field->InsertNextValue(error);

			auto reset_pt = test_reset.ReadPortal().Get(n).Pos;
			test_set.WritePortal().Set(n, vtkm::Particle(Vec3f(static_cast<vtkm::FloatDefault>(reset_pt[0]), 
																														 static_cast<vtkm::FloatDefault>(reset_pt[1]), 
																														 static_cast<vtkm::FloatDefault>(0.0)), n));
		}
		
		vtkSmartPointer<vtkRectilinearGrid> extraGrid = 
			vtkSmartPointer<vtkRectilinearGrid>::New();
		
		extraGrid->SetDimensions(sdims[0], sdims[1], 1);
  	extraGrid->SetXCoordinates(sxCoords);
  	extraGrid->SetYCoordinates(syCoords);
  	extraGrid->SetZCoordinates(szCoords);

		extraGrid->GetCellData()->AddArray(endpt_cell_field);
		vtkSmartPointer<vtkCellDataToPointData> c2p = vtkSmartPointer<vtkCellDataToPointData>::New();
	
		c2p->AddInputData(extraGrid);
		c2p->Update();
	
		vtkAbstractArray* endpt_vals0 = c2p->GetOutput()->GetPointData()->GetArray(0);
	  vtkFloatArray* endpt_point_field = vtkFloatArray::SafeDownCast(endpt_vals0);
		std::stringstream s_e_pt;
		s_e_pt << "error_" << argv[2] << "_" << i;
		endpt_point_field->SetName(s_e_pt.str().c_str());
 
    vtkm::cont::ArrayHandle<vtkm::FloatDefault> ftleField;
    using AnalysisType = vtkm::worklet::LagrangianStructures<2>;
    AnalysisType ftleCalculator(advectionTime, cells2d);
    vtkm::worklet::DispatcherMapField<AnalysisType> dispatcher(ftleCalculator);
    dispatcher.Invoke(lcsInputPoints, lcsOutputPoints, ftleField);
		
		vtkSmartPointer<vtkFloatArray> ftle_field = vtkSmartPointer<vtkFloatArray>::New();
		std::stringstream s_f;
		s_f << "ftle_" << argv[2] << "_" << i;
		ftle_field->SetName(s_f.str().c_str());	

		for(int i = 0; i < num_seeds_interval; i++)
		{
			auto f = ftleField.ReadPortal().Get(i);
			ftle_field->InsertNextValue(f);
			ftle_val[i] = f*1.0;
			error_val[i] = endpt_point_field->GetTuple1(i)*1.0;

		}	
		
		cnpy::npz_save("out.npz", s_f.str().c_str(), &ftle_val[0], {sdims[0], sdims[1], 1}, "a");	
		cnpy::npz_save("out.npz", s_e_pt.str().c_str(), &error_val[0], {sdims[0], sdims[1], 1}, "a");	


/*	
		vtkSmartPointer<vtkRectilinearGrid> outputGrid = 
			vtkSmartPointer<vtkRectilinearGrid>::New();
		
		outputGrid->SetDimensions(sdims[0], sdims[1], 1);
  	outputGrid->SetXCoordinates(sxCoords);
  	outputGrid->SetYCoordinates(syCoords);
  	outputGrid->SetZCoordinates(szCoords);
		outputGrid->GetPointData()->AddArray(ftle_field);
//		outputGrid->GetPointData()->AddArray(displacement_field);
//		outputGrid->GetCellData()->AddArray(endpt_cell_field);
		outputGrid->GetPointData()->AddArray(endpt_point_field);
		
  	std::stringstream op;
  	op << output_file << "_" << i <<  ".vtk";

  	writer->SetFileName(op.str().c_str());
  	writer->SetInputData(outputGrid);
  	writer->SetFileTypeToASCII();
  	writer->Write();
*/
	}

}

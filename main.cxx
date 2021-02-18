#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkDataArray.h>
#include <vtkSmartPointer.h>
#include <vtkRectilinearGridReader.h>
#include <vtkRectilinearGrid.h>
#include <vtkStructuredGrid.h>
#include <vtkGradientFilter.h>
#include <vtkDataSet.h>
#include <sstream>
#include <iostream>
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


#include <vtkm/worklet/lcs/GridMetaData.h>
#include <vtkm/worklet/lcs/LagrangianStructureHelpers.h>

#include <vtkm/worklet/LagrangianStructures.h>

int main(int argc, char* argv[])
{
/*
 * Load a file
 * Get dims, bounds
 * Generate seed_0 --- locations of seed particles at time 0
 * Extract the 2D slice at time 0
 * Compute entropy, divergence, vorticitymagnitude 
 * flow maps
 * ftle 
 */

	std::stringstream s;
	s << argv[1] << argv[2] << argv[3];
	std::cout << "Inputfile: " << s.str() << std::endl;

  vtkSmartPointer<vtkRectilinearGridReader> reader =
    vtkSmartPointer<vtkRectilinearGridReader>::New();
  reader->SetFileName(s.str().c_str());
  reader->Update();

  vtkSmartPointer<vtkRectilinearGrid> mesh =
    vtkSmartPointer<vtkRectilinearGrid>::New();
  mesh = reader->GetOutput();
  int num_pts = mesh->GetNumberOfPoints();
	int dims[3];
	dims[0] = 512;
	dims[1] = 512;
	dims[2] = 1001;

	int num_pts_slice = dims[0]*dims[1];	
//  std::cout << "The number of points: " << num_pts << std::endl;

  vtkAbstractArray* a1 = mesh->GetPointData()->GetArray("u");
  vtkAbstractArray* a2 = mesh->GetPointData()->GetArray("v");

  vtkFloatArray* att1 = vtkFloatArray::SafeDownCast(a1);
  vtkFloatArray* att2 = vtkFloatArray::SafeDownCast(a2);

  float *xc = (float*) mesh->GetXCoordinates()->GetVoidPointer(0);
  float *yc = (float*) mesh->GetYCoordinates()->GetVoidPointer(0);
  float *zc = (float*) mesh->GetZCoordinates()->GetVoidPointer(0);

	float x_spacing = (xc[dims[0]-1] - xc[0])/(dims[0]-1);
	float y_spacing = (yc[dims[1]-1] - yc[0])/(dims[1]-1);
	float z_spacing = (zc[dims[2]-1] - zc[0])/(dims[2]-1);	

/* Extract 2D slice */ 

	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> vel_array = vtkSmartPointer<vtkDoubleArray>::New();
	vel_array->SetNumberOfComponents(3);
	vel_array->SetNumberOfTuples(num_pts_slice);
	vel_array->SetName("velocity");

	for(int j = 0; j < dims[1]; j++)
	{
		for(int i = 0; i < dims[0]; i++)
		{
			int index = j*dims[0] + i;
			double vec[3];
			vec[0] = att1->GetTuple1(index);
			vec[1] = att2->GetTuple1(index);
			vec[2] = 0.0;
			points->InsertNextPoint(xc[i], yc[j], 0);
			vel_array->InsertTuple(index, vec);	
		}
	}	

	vtkSmartPointer<vtkStructuredGrid> slice = vtkSmartPointer<vtkStructuredGrid>::New();
	slice->SetDimensions(dims[0], dims[1], 1);
	slice->SetPoints(points);
	slice->GetPointData()->AddArray(vel_array);

	vtkSmartPointer<vtkGradientFilter> gradient = vtkSmartPointer<vtkGradientFilter>::New();	
	gradient->SetInputData(slice);
	gradient->SetInputScalars(vtkDataObject::FIELD_ASSOCIATION_POINTS, "velocity");
	gradient->SetComputeVorticity(1);
	gradient->SetComputeQCriterion(1);
	gradient->SetComputeDivergence(1);
	gradient->Update();

  vtkDoubleArray *gradient_field = vtkDoubleArray::SafeDownCast(gradient->GetOutput()->GetPointData()->GetArray("Gradients"));
  vtkDoubleArray *vorticity_field = vtkDoubleArray::SafeDownCast(gradient->GetOutput()->GetPointData()->GetArray("Vorticity"));
  vtkDoubleArray *qcriterion_field = vtkDoubleArray::SafeDownCast(gradient->GetOutput()->GetPointData()->GetArray("Q-criterion"));
  vtkDoubleArray *divergence_field = vtkDoubleArray::SafeDownCast(gradient->GetOutput()->GetPointData()->GetArray("Divergence"));

	std::cout << "Gradient field: " << gradient_field->GetNumberOfValues() << " , " << gradient_field->GetNumberOfComponents() << std::endl;
	std::cout << "Vorticity field: " << vorticity_field->GetNumberOfValues() << " , " << vorticity_field->GetNumberOfComponents() << std::endl;
	std::cout << "Q-criterion field: " << qcriterion_field->GetNumberOfValues() << " , " << qcriterion_field->GetNumberOfComponents() << std::endl;
	std::cout << "Divergence field: " << divergence_field->GetNumberOfValues() << " , " << divergence_field->GetNumberOfComponents() << std::endl;

	/*
 *  For a grid of starting locations S in three dimensions.
 *  I want to compute where these particles travel after time = 1 and 2, with a step size of 0.01, 100 and 200 steps 
 *  I want to store the locations at t1 and the corresponding ftle. 
 *  Then I also want to store  
*/  

	/*
 * Set up to compute flow maps, and two sets of long trajectories. 
 */

	float* uni_x = (float*)malloc(sizeof(float)*num_pts_slice);
	float* uni_y = (float*)malloc(sizeof(float)*num_pts_slice); 
	vtkm::cont::ArrayHandle<vtkm::Particle> start_set, current_set;
	start_set.Allocate(num_pts_slice);
	current_set.Allocate(num_pts_slice);
	 using Vec3f = vtkm::Vec<vtkm::FloatDefault, 3>;		
  	
	vtkm::cont::ArrayHandle<vtkm::Vec2f> lcsInputPoints, lcsOutputPoints;
	lcsInputPoints.Allocate(dims[0]*dims[1]);		
	lcsOutputPoints.Allocate(dims[0]*dims[1]);
	
	for(int j = 0; j < dims[1]; j++)
	{
		for(int i = 0; i < dims[0]; i++)
		{
			int index = j*dims[0] + i;
			uni_x[index] = xc[0] + i*x_spacing;
			uni_y[index] = yc[0] + j*y_spacing;
			start_set.WritePortal().Set(index, vtkm::Particle(Vec3f(static_cast<vtkm::FloatDefault>(uni_x[index]), static_cast<vtkm::FloatDefault>(uni_y[index]), static_cast<vtkm::FloatDefault>(0.0)), index));
			current_set.WritePortal().Set(index, vtkm::Particle(Vec3f(static_cast<vtkm::FloatDefault>(uni_x[index]), static_cast<vtkm::FloatDefault>(uni_y[index]), static_cast<vtkm::FloatDefault>(0.0)), index));
			lcsInputPoints.WritePortal().Set(index, vtkm::Vec2f(static_cast<vtkm::FloatDefault>(uni_x[index]), static_cast<vtkm::FloatDefault>(uni_y[index])));
			lcsOutputPoints.WritePortal().Set(index, vtkm::Vec2f(static_cast<vtkm::FloatDefault>(uni_x[index]), static_cast<vtkm::FloatDefault>(uni_y[index])));
		}
	}

/* Create the 3D data set where the Z dimension is time. */
	
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
// This dataset object can be used to compute all the ftle_set pathlines


	vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> velocity_field;
	velocity_field.Allocate(num_pts);
	for(int i = 0; i < num_pts; i++)
	{
		velocity_field.WritePortal().Set(i, vtkm::Vec<vtkm::FloatDefault, 3>(att1->GetTuple1(i), att2->GetTuple1(i), 1.0));
	}

/* Create the particle advection worklet set up */

	using FieldHandle = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>>;	
	const vtkm::cont::DynamicCellSet& cells = dataset.GetCellSet();
	const vtkm::cont::CoordinateSystem& coords = dataset.GetCoordinateSystem();
	using GridEvalType = vtkm::worklet::particleadvection::GridEvaluator<FieldHandle>;
	
	vtkm::worklet::ParticleAdvection particleadvection;
	vtkm::worklet::ParticleAdvectionResult res;

	using RK4Type = vtkm::worklet::particleadvection::RK4Integrator<GridEvalType>;
	GridEvalType eval_ftle(coords, cells, velocity_field);
	RK4Type rk4(eval_ftle, static_cast<vtkm::Float32>(0.01));
	
/* Set up FTLE computation */

	vtkm::cont::DataSet lcsInput;
	vtkm::Id2 lcsGridDims(dims[0], dims[1]);	
	vtkm::Vec2f origin2d(static_cast<vtkm::FloatDefault>(xc[0]),
									 static_cast<vtkm::FloatDefault>(yc[0]));
	vtkm::Vec2f spacing2d(static_cast<vtkm::FloatDefault>(x_spacing),
										static_cast<vtkm::FloatDefault>(y_spacing));
	vtkm::cont::DataSetBuilderUniform uniformDatasetBuilder2d;
	lcsInput = uniformDatasetBuilder2d.Create(lcsGridDims, origin2d, spacing2d);
	vtkm::cont::DynamicCellSet lcsCellSet = lcsInput.GetCellSet();
		
/* Set up output -- need to update to work for VTK better */

  vtkm::cont::DataSet output;
  output.AddCoordinateSystem(lcsInput.GetCoordinateSystem());
  output.SetCellSet(lcsInput.GetCellSet());

	for(int c = 1; c <= 10; c++)
	{
		res = particleadvection.Run(rk4, current_set, 100);

		// Use res to update the current_set and lcsOutputPoints
		auto updated_particles = res.Particles;
		for(int i = 0; i < num_pts_slice; i++)
		{
			auto pt = updated_particles.ReadPortal().Get(i).Pos;
			current_set.WritePortal().Set(i, vtkm::Particle(Vec3f(static_cast<vtkm::FloatDefault>(pt[0]), 
																														static_cast<vtkm::FloatDefault>(pt[1]), 
																														static_cast<vtkm::FloatDefault>(pt[2])), i));
			lcsOutputPoints.WritePortal().Set(i, vtkm::Vec2f(static_cast<vtkm::FloatDefault>(pt[0]), 
																														static_cast<vtkm::FloatDefault>(pt[1])));
		}
	
		vtkm::FloatDefault advectionTime = c*1.0f;
  	vtkm::cont::ArrayHandle<vtkm::FloatDefault> outputField;
		using AnalysisType = vtkm::worklet::LagrangianStructures<2>;
  	AnalysisType ftleCalculator(advectionTime, lcsCellSet);
  	vtkm::worklet::DispatcherMapField<AnalysisType> dispatcher(ftleCalculator);
	  dispatcher.Invoke(lcsInputPoints, lcsOutputPoints, outputField);
		std::stringstream field_name;
		field_name << "ftle_" << c;
  	output.AddPointField(field_name.str().c_str(), outputField);
		std::cout << "Completed interval : " << c << std::endl;
	}
	
	vtkm::io::VTKDataSetWriter writer("ftle.vtk");
	writer.WriteDataSet(output);

/*  vector<vtkm::Float64> temp_d, temp_v, temp_q, temp_g;

  for(int i = 0; i < divergence_field->GetNumberOfValues(); i++)
    temp_d.push_back(divergence_field->GetValue(i));

  for(int i = 0; i < vorticity_field->GetNumberOfValues(); i++)
  { 
    double vorticity[3];
    vorticity_field->GetTuple(i,vorticity);
    temp_v.push_back(vtkm::make_Vec(vorticity[0],vorticity[1],vorticity[2]));
  }
*/
}
